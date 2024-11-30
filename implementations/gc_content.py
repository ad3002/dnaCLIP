from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class GCDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # First handle special fields
        gc_values = [f.pop("gc_content") for f in features]
        
        # Then do the normal collation
        batch = super().__call__(features)
        
        # Add back the gc_content as labels
        batch["labels"] = torch.tensor(gc_values, dtype=torch.float32)
        return batch

class GcContentDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.data_collator = None
    
    def generate_features(self, sequence):
        gc_count = sum(1 for base in sequence.upper() if base in ['G', 'C'])
        return gc_count / len(sequence) if sequence else 0.0
    
    def prepare_dataset(self, dataset, tokenizer):
        def preprocess_function(examples):
            # First tokenize
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors=None
            )
            
            # Calculate and add GC content
            tokenized['gc_content'] = [
                self.generate_features(seq) for seq in examples["sequence"]
            ]
            
            # Convert to format that plays nice with datasets
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "gc_content": tokenized["gc_content"]
            }
            
        # Create custom data collator
        self.data_collator = GCDataCollator(tokenizer=tokenizer)
        
        # Process each split
        processed_dataset = {}
        for split in dataset.keys():
            processed_dataset[split] = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Processing {split} split"
            )
            
        return processed_dataset

class GcContentHead(BaseHead):
    def __init__(self, input_dim=768):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequence_features, **kwargs):
        return self.regressor(sequence_features)
    
    def compute_loss(self, outputs, targets):
        return F.mse_loss(outputs.squeeze(), targets.float())

class GcContentTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        if 'data_collator' not in kwargs and 'model' in kwargs:
            kwargs['data_collator'] = kwargs['model'].data_generator.data_collator
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        try:
            # Convert inputs to dict if it's not already
            if not isinstance(inputs, dict):
                inputs = inputs.data
                
            # Get labels, trying different possible keys
            if "labels" in inputs:
                labels = inputs["labels"]
            elif "gc_content" in inputs:
                labels = inputs["gc_content"]
            else:
                raise KeyError("No labels found in inputs")
                
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            loss = model.head.compute_loss(outputs, labels)
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            print(f"Input keys available: {inputs.keys()}")
            raise e

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step to ensure consistent shapes"""
        inputs = self._prepare_inputs(inputs)
        gc_content = inputs["gc_content"] if "gc_content" in inputs else inputs["labels"]
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            loss = model.head.compute_loss(outputs, gc_content)
            
        return (loss.detach(), outputs.detach(), gc_content)

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        mse = F.mse_loss(torch.tensor(predictions), torch.tensor(labels))
        correlation = torch.corrcoef(torch.stack([
            torch.tensor(predictions).flatten(),
            torch.tensor(labels).flatten()
        ]))[0,1]
        return {
            'mse': mse.item(),
            'correlation': correlation.item()
        }

# Register implementation
DNAModelRegistry.register(
    "gc_content",
    GcContentHead,
    GcContentDataGenerator,
    GcContentTrainer
)