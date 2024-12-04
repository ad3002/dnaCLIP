from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import DataCollatorWithPadding, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class BendabilityDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        bend_values = []
        original_sequences = []
        filtered_features = []
        
        for f in features:
            feature_copy = dict(f)
            if "bendability" in feature_copy:
                bend_values.append(torch.tensor(feature_copy["bendability"]))
            if "original_sequence" in feature_copy:
                original_sequences.append(feature_copy.pop("original_sequence"))
            filtered_features.append(feature_copy)
        
        batch = super().__call__(filtered_features)
        if bend_values:
            batch["labels"] = torch.stack(bend_values)
        if original_sequences:
            batch["original_sequence"] = original_sequences
        return batch

class BendabilityDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        super().__init__()  # Add parent class initialization
        self.max_length = max_length
        self.data_collator = None
        
        # DNAse I-based bendability parameters
        # Values from Brukner et al. and other experimental sources
        self.bendability_params = {
            'AAA': 0.14, 'AAT': 0.16, 'AAG': 0.18, 'AAC': 0.17,
            'ATA': 0.15, 'ATT': 0.14, 'ATG': 0.16, 'ATC': 0.15,
            'AGA': 0.17, 'AGT': 0.16, 'AGG': 0.18, 'AGC': 0.17,
            'ACA': 0.16, 'ACT': 0.15, 'ACG': 0.17, 'ACC': 0.16,
            'TAA': 0.15, 'TAT': 0.14, 'TAG': 0.16, 'TAC': 0.15,
            'TTA': 0.14, 'TTT': 0.14, 'TTG': 0.16, 'TTC': 0.15,
            'TGA': 0.16, 'TGT': 0.15, 'TGG': 0.17, 'TGC': 0.16,
            'TCA': 0.15, 'TCT': 0.14, 'TCG': 0.16, 'TCC': 0.15,
            'GAA': 0.18, 'GAT': 0.16, 'GAG': 0.19, 'GAC': 0.17,
            'GTA': 0.16, 'GTT': 0.15, 'GTG': 0.17, 'GTC': 0.16,
            'GGA': 0.18, 'GGT': 0.17, 'GGG': 0.19, 'GGC': 0.18,
            'GCA': 0.17, 'GCT': 0.16, 'GCG': 0.18, 'GCC': 0.17,
            'CAA': 0.17, 'CAT': 0.15, 'CAG': 0.17, 'CAC': 0.16,
            'CTA': 0.15, 'CTT': 0.14, 'CTG': 0.16, 'CTC': 0.15,
            'CGA': 0.17, 'CGT': 0.16, 'CGG': 0.18, 'CGC': 0.17,
            'CCA': 0.16, 'CCT': 0.15, 'CCG': 0.17, 'CCC': 0.16
        }
    
    def calculate_bendability(self, sequence):
        """Calculate average bendability score"""
        sequence = sequence.upper()
        if len(sequence) < 3:
            return 0.0
            
        bendability_sum = 0.0
        count = 0
        
        for i in range(len(sequence) - 2):
            trinuc = sequence[i:i+3]
            if trinuc in self.bendability_params:
                bendability_sum += self.bendability_params[trinuc]
                count += 1
        
        return bendability_sum / count if count > 0 else 0.0
    
    def prepare_dataset(self, dataset, tokenizer):
        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            bend_values = [
                [self.calculate_bendability(seq)] for seq in examples["sequence"]
            ]
            
            tokenized["bendability"] = bend_values
            tokenized["original_sequence"] = examples["sequence"]
            
            return tokenized
        
        self.data_collator = BendabilityDataCollator(tokenizer=tokenizer)
        
        processed_dataset = {}
        for split in dataset.keys():
            processed_dataset[split] = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Processing {split} split"
            )
        
        return processed_dataset

    def generate_features(self, sequence):
        """Required abstract method implementation"""
        if not sequence:
            return [0.0]
        return [self.calculate_bendability(sequence)]

class BendabilityHead(BaseHead):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Single output for bendability
            nn.Sigmoid()
        )
    
    def forward(self, sequence_features, **kwargs):
        return self.regressor(sequence_features)
    
    def compute_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets.float())
    
    def test(self, sequence_features, **kwargs):
        """Required abstract method implementation"""
        with torch.no_grad():
            return self.forward(sequence_features, **kwargs)

class BendabilityTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get('args', None)
        if training_args:
            training_args.label_names = ["bendability"]
        
        # Handle both tokenizer and processing_class
        self.processing_class = kwargs.pop('processing_class', None) or kwargs.get('tokenizer', None)
        
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
            
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Ensure proper shapes and types
        predictions = np.asarray(predictions).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        
        # Ensure equal lengths
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]
        
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        return {
            'mse': mse,
            'mae': mae,
        }

    def test_model(self, test_dataset, num_examples=10):
        model = self.model.eval()
        all_predictions = []
        all_labels = []
        all_sequences = []
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size if self.args else 8,
            collate_fn=self.data_collator
        )
        
        device = next(model.parameters()).device
        
        # More robust processor selection
        processor = (self.processing_class or 
                    getattr(self, 'tokenizer', None) or 
                    getattr(self.model, 'processing_class', None) or 
                    getattr(self.model, 'tokenizer', None))
        
        if not processor:
            raise ValueError("No processing_class or tokenizer available. Please provide one when initializing the trainer.")
        
        for batch in dataloader:
            # Filter inputs to only include required keys
            filtered_inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            
            with torch.no_grad():
                outputs = model(**filtered_inputs)
                predictions = outputs.cpu().numpy()
            
            labels = batch['labels'].cpu().numpy()
            sequences = [processor.decode(seq, skip_special_tokens=True) 
                       for seq in batch['input_ids'].cpu().numpy()]
            
            all_predictions.extend(predictions.reshape(-1))
            all_labels.extend(labels.reshape(-1))
            all_sequences.extend(sequences)
        
        metrics = self.compute_metrics((np.array(all_predictions), np.array(all_labels)))
        
        print("\nTest Results:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        if num_examples > 0:
            print("\nSample Predictions:")
            print("Sequence\t\tPredicted\tActual\t\tDiff")
            print("-" * 80)
            
            indices = np.random.choice(len(all_predictions), 
                                     min(num_examples, len(all_predictions)), 
                                     replace=False)
            for idx in indices:
                seq = all_sequences[idx]
                pred = all_predictions[idx]
                actual = all_labels[idx]
                diff = abs(pred - actual)
                print(f"{seq[:20]}...\t{pred:.3f}\t\t{actual:.3f}\t\t{diff:.3f}")
        
        return {
            **metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'sequences': all_sequences
        }

    def get_test_metrics(self, predictions, labels):
        """Required abstract method implementation"""
        return self.compute_metrics((predictions, labels))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not isinstance(inputs, dict):
            inputs = {k: v for k, v in inputs.items()}
            
        labels = inputs.get("labels", None)
        if labels is None:
            labels = inputs.get("bendability", None)
        if labels is None:
            raise KeyError(f"No labels found in inputs. Available keys: {inputs.keys()}")
        
        # Ensure labels have correct shape and type
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 1:
                labels = labels.view(-1, 1)
            labels = labels.float()
            
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Ensure outputs have correct shape
        if outputs.dim() == 1:
            outputs = outputs.view(-1, 1)
        
        loss = model.head.compute_loss(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss

def test_bendability_implementation(model, test_dataset, tokenizer, num_examples=10):
    """Standalone test function for bendability prediction"""
    trainer = BendabilityTrainer(
        model=model,
        args=None,
        tokenizer=tokenizer,  # Change back to tokenizer for proper handling
        processing_class=tokenizer  # Add both for compatibility
    )
    trainer.processing_class = tokenizer  # Ensure it's set
    trainer.data_collator = BendabilityDataCollator(tokenizer=tokenizer)
    return trainer.test_model(test_dataset, num_examples)

# Register the bendability implementation
DNAModelRegistry.register(
    "bendability",
    BendabilityHead,
    BendabilityDataGenerator,
    BendabilityTrainer,
    test_bendability_implementation
)