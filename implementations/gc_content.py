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
        batch = super().__call__(features)
        
        # Convert gc_content to tensor
        if "gc_content" in features[0]:
            batch["labels"] = torch.tensor([f["gc_content"] for f in features], dtype=torch.float32)
            
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
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors=None
            )
            # Calculate GC content
            gc_content = [self.generate_features(seq) for seq in examples["sequence"]]
            tokenized['gc_content'] = gc_content
            return tokenized
            
        # Create custom data collator
        self.data_collator = GCDataCollator(tokenizer=tokenizer)
        
        # Process each split
        processed_dataset = {}
        for split in dataset.keys():
            processed_dataset[split] = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=['Unnamed: 0', 'sequence', 'promoter_presence']
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
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')  # Changed from gc_content to labels
        if labels is None:
            raise ValueError(f"No labels found in inputs. Keys: {inputs.keys()}")
            
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        loss = model.head.compute_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss

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