from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import DataCollatorWithPadding, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List, Union

NUCLEOTIDE_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
IDX_TO_NUCLEOTIDE = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
COMPLEMENT = {
    'A': 'T', 
    'T': 'A', 
    'G': 'C', 
    'C': 'G',
    'N': 'N',  # Non-standard nucleotide
    'R': 'Y',  # Purine (A or G)
    'Y': 'R',  # Pyrimidine (C or T)
    'K': 'M',  # Keto (G or T)
    'M': 'K',  # Amino (A or C)
    'S': 'S',  # Strong (G or C)
    'W': 'W',  # Weak (A or T)
    'B': 'V',  # not A (C or G or T)
    'V': 'B',  # not T (A or C or G)
    'D': 'H',  # not C (A or G or T)
    'H': 'D',  # not G (A or C or T)
}

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
        """Convert sequence to numerical labels with handling for non-standard nucleotides"""
        return [NUCLEOTIDE_TO_IDX.get(nt, NUCLEOTIDE_TO_IDX['N']) for nt in sequence.upper()]
    
    def get_reverse_complement(self, sequence):
        """Get reverse complement with handling for non-standard nucleotides"""
        return ''.join(COMPLEMENT.get(nt, 'N') for nt in sequence.upper()[::-1])
    
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
            nn.Linear(256, 5)  # 5 classes for A,T,G,C,N
        )
    
    def forward(self, sequence_features, **kwargs):
        """Forward pass handling both sequence-level and token-level features"""
        # Handle both cases: [batch_size, hidden_dim] and [batch_size, seq_length, hidden_dim]
        if len(sequence_features.shape) == 2:
            batch_size, hidden_dim = sequence_features.shape
            # Assume single token for sequence-level features
            logits = self.classifier(sequence_features)  # [batch_size, num_classes]
            return logits.unsqueeze(1)  # [batch_size, 1, num_classes]
        else:
            batch_size, seq_length, _ = sequence_features.shape
            # Reshape for efficiency
            flat_features = sequence_features.view(-1, sequence_features.size(-1))
            logits = self.classifier(flat_features)
            return logits.view(batch_size, seq_length, -1)
    
    def compute_loss(self, outputs, targets):
        """Compute cross entropy loss for any input shape"""
        num_classes = outputs.size(-1)
        return F.cross_entropy(
            outputs.view(-1, num_classes),
            targets.view(-1)
        )
    
    def test(self, sequence_features, **kwargs):
        """Test method for reverse complement prediction"""
        with torch.no_grad():
            return self.forward(sequence_features, **kwargs)

class ReverseComplementTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get('args', None)
        if training_args:
            # Ensure label_names is set correctly
            training_args.label_names = ["rev_comp_labels"]
            
        # Set compute_metrics before parent initialization
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
            
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Convert inputs to dict if it's not already
        if not isinstance(inputs, dict):
            inputs = {k: v for k, v in inputs.items()}
        
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Get labels
        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError(f"No labels found in inputs. Keys: {inputs.keys()}")
        
        # Compute loss
        loss = model.head.compute_loss(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs):
        """Prepare inputs before compute_loss is called"""
        prepared = super()._prepare_inputs(inputs)
        if isinstance(inputs, dict) and "rev_comp_labels" in inputs:
            prepared["labels"] = inputs["rev_comp_labels"]
        return prepared

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

# Register implementation
DNAModelRegistry.register(
    "reverse_complement",
    ReverseComplementHead,
    ReverseComplementDataGenerator,
    ReverseComplementTrainer
)