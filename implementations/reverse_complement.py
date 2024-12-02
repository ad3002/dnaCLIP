from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import DataCollatorWithPadding, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List, Union

NUCLEOTIDE_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
IDX_TO_NUCLEOTIDE = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

@dataclass
class ReverseComplementDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Collect reverse complement labels if available
        rev_comp_labels = [f["rev_comp_labels"] for f in features if "rev_comp_labels" in f]
        
        # Proceed with default collation
        batch = super().__call__(features)
        
        # Add labels if collected
        if rev_comp_labels:
            batch["labels"] = torch.tensor(rev_comp_labels, dtype=torch.long)
        
        return batch

class ReverseComplementDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.data_collator = None
    
    def generate_features(self, sequence):
        """Generate reverse complement features for a single sequence"""
        # For reverse complement task, we return the sequence itself 
        # since the actual processing happens in prepare_dataset
        return sequence
    
    def sequence_to_labels(self, sequence):
        # Convert sequence to numerical labels
        return [NUCLEOTIDE_TO_IDX[nt] for nt in sequence.upper()]
    
    def get_reverse_complement(self, sequence):
        return ''.join(COMPLEMENT[nt] for nt in sequence.upper()[::-1])
    
    def prepare_dataset(self, dataset, tokenizer):
        def preprocess_function(examples):
            # Tokenize sequences
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Get reverse complement sequences and convert to labels
            rev_comp_seqs = [
                self.get_reverse_complement(seq) for seq in examples["sequence"]
            ]
            rev_comp_labels = [
                self.sequence_to_labels(seq) for seq in rev_comp_seqs
            ]
            
            tokenized["rev_comp_labels"] = rev_comp_labels
            tokenized["original_sequence"] = examples["sequence"]
            
            return tokenized
                
        self.data_collator = ReverseComplementDataCollator(tokenizer=tokenizer)
        
        processed_dataset = {}
        for split in dataset.keys():
            processed_dataset[split] = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Processing {split} split"
            )
                
        return processed_dataset

class ReverseComplementHead(BaseHead):
    def __init__(self, input_dim=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # 4 classes for A,T,G,C
        )
    
    def forward(self, sequence_features, **kwargs):
        # Output shape: [batch_size, seq_length, 4]
        return self.classifier(sequence_features)
    
    def compute_loss(self, outputs, targets):
        # Reshape for CrossEntropyLoss
        outputs = outputs.view(-1, 4)  # [batch_size * seq_length, 4]
        targets = targets.view(-1)      # [batch_size * seq_length]
        return F.cross_entropy(outputs, targets)
    
    def test(self, sequence_features, **kwargs):
        """Test method for reverse complement prediction"""
        with torch.no_grad():
            return self.forward(sequence_features, **kwargs)

class ReverseComplementTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def get_default_args(output_dir: str) -> TrainingArguments:
        args = BaseTrainer.get_default_args(output_dir)
        args.label_names = ["rev_comp_labels"]
        args.metric_for_best_model = "eval_sequence_accuracy"
        args.greater_is_better = True
        return args
    
    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(-1)
        
        # Calculate position-wise accuracy
        position_accuracy = (predictions == labels).mean()
        
        # Calculate sequence-wise accuracy
        sequence_matches = (predictions == labels).all(axis=1)
        sequence_accuracy = sequence_matches.mean()
        
        return {
            'position_accuracy': float(position_accuracy),
            'sequence_accuracy': float(sequence_accuracy)
        }

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs["rev_comp_labels"]
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            loss = model.head.compute_loss(outputs, labels)
            
        return (loss.detach(), outputs.detach(), labels)

# Register implementation
DNAModelRegistry.register(
    "reverse_complement",
    ReverseComplementHead,
    ReverseComplementDataGenerator,
    ReverseComplementTrainer
)