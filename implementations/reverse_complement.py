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
        self.dense = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, 5)  # 5 classes for A,T,G,C,N
    
    def forward(self, sequence_features, attention_mask=None, **kwargs):
        """Process sequence-level features and output per-position predictions"""
        # Handle the case when sequence_features is from pooled output
        if len(sequence_features.shape) == 2:
            if attention_mask is None:
                raise ValueError("attention_mask required for sequence length reconstruction")
            sequence_length = attention_mask.shape[1]
            sequence_features = sequence_features.unsqueeze(1).expand(-1, sequence_length, -1)
        
        # Process features
        sequence_features = self.dense(sequence_features)
        sequence_features = self.layer_norm(sequence_features)
        logits = self.classifier(sequence_features)  # [batch_size, seq_length, num_classes]
        
        return logits
    
    def compute_loss(self, outputs, targets):
        """Compute cross entropy loss with shape checking"""
        if len(outputs.shape) == 2:
            # Handle case where outputs are [batch_size, num_classes]
            outputs = outputs.unsqueeze(1)  # [batch_size, 1, num_classes]
        
        batch_size, seq_length, num_classes = outputs.shape
        if targets.shape[1] != seq_length:
            targets = targets[:, :seq_length]
        
        outputs = outputs.contiguous().view(-1, num_classes)
        targets = targets.contiguous().view(-1)
        
        return F.cross_entropy(outputs, targets)
    
    def test(self, sequence_features, attention_mask=None, **kwargs):
        """Test method for reverse complement prediction"""
        with torch.no_grad():
            # Forward pass with no gradient computation
            logits = self.forward(sequence_features, attention_mask=attention_mask)
            predictions = logits.argmax(dim=-1)
            return predictions

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
        """Compute loss with proper shape handling"""
        labels = inputs.get("labels", inputs.get("rev_comp_labels"))
        if labels is None:
            raise ValueError(f"No labels found in inputs. Keys: {inputs.keys()}")
        
        # Get model outputs
        backbone_output = model.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True
        )
        
        # Handle different types of backbone outputs
        if hasattr(backbone_output, 'hidden_states'):
            sequence_features = backbone_output.hidden_states[-1]
        elif hasattr(backbone_output, 'last_hidden_state'):
            sequence_features = backbone_output.last_hidden_state
        elif isinstance(backbone_output, dict):
            sequence_features = backbone_output.get('last_hidden_state', 
                                                backbone_output.get('hidden_states', [-1])[-1])
        elif isinstance(backbone_output, tuple):
            sequence_features = backbone_output[0]
        else:
            # For MaskedLMOutput, use the hidden states
            sequence_features = backbone_output[2][-1]  # Get last hidden state from hidden_states tuple
        
        # Get predictions from head
        outputs = model.head(sequence_features, attention_mask=inputs["attention_mask"])
        
        # Ensure outputs and labels have matching sequence lengths
        seq_length = outputs.shape[1]
        labels = labels[:, :seq_length]
        
        # Compute loss
        loss = model.head.compute_loss(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs):
        """Prepare inputs ensuring label consistency"""
        prepared = super()._prepare_inputs(inputs)
        if isinstance(inputs, dict):
            if "rev_comp_labels" in inputs:
                # Keep original shape of labels
                prepared["labels"] = inputs["rev_comp_labels"]
        return prepared

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step with shape handling"""
        inputs = self._prepare_inputs(inputs)
        labels = inputs["rev_comp_labels"]
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # Ensure outputs and labels have matching sequence lengths
            seq_length = outputs.shape[1]
            labels = labels[:, :seq_length]
            
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

    def test_model(self, test_dataset, num_examples=10):
        """Test reverse complement prediction model with proper sequence handling"""
        model = self.model.eval()
        all_predictions = []
        all_labels = []
        all_sequences = []
        device = next(model.parameters()).device
        
        for i in range(0, len(test_dataset), self.args.per_device_eval_batch_size):
            batch = test_dataset[i:i+self.args.per_device_eval_batch_size]
            
            # Create list of features for the data collator
            features = [{
                'input_ids': batch['input_ids'][j],
                'attention_mask': batch['attention_mask'][j],
                'rev_comp_labels': batch['rev_comp_labels'][j]
            } for j in range(len(batch['input_ids']))]
            
            # Use data collator to properly pad the batch
            padded_batch = self.data_collator(features)
            
            # Move tensors to device
            inputs = {
                "input_ids": padded_batch["input_ids"].to(device),
                "attention_mask": padded_batch["attention_mask"].to(device),
                "rev_comp_labels": padded_batch["labels"].to(device)
            }
            
            with torch.no_grad():
                # Get backbone outputs
                backbone_output = model.backbone(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True
                )
                
                # Handle different types of backbone outputs
                if hasattr(backbone_output, 'hidden_states'):
                    sequence_features = backbone_output.hidden_states[-1]
                elif hasattr(backbone_output, 'last_hidden_state'):
                    sequence_features = backbone_output.last_hidden_state
                elif isinstance(backbone_output, dict):
                    sequence_features = backbone_output.get('last_hidden_state', 
                                                        backbone_output.get('hidden_states', [-1])[-1])
                elif isinstance(backbone_output, tuple):
                    sequence_features = backbone_output[0]
                else:
                    # For MaskedLMOutput, use the hidden states
                    sequence_features = backbone_output[2][-1]  # Get last hidden state from hidden_states tuple
                
                # Get head outputs
                outputs = model.head(sequence_features)
                predictions = outputs.argmax(dim=-1)
            
            # Get sequences for display
            sequences = [
                self.processing_class.decode(seq, skip_special_tokens=True) 
                for seq in batch["input_ids"]
            ]
            
            # Process each sequence in the batch
            for idx in range(len(predictions)):
                # Get attention mask for current sequence
                mask = inputs["attention_mask"][idx].bool()
                seq_len = mask.sum().item()
                
                # Get actual sequence length predictions and labels
                pred = predictions[idx, :seq_len].cpu().numpy()
                label = inputs["rev_comp_labels"][idx, :seq_len].cpu().numpy()
                
                all_predictions.append(pred)
                all_labels.append(label)
            
            all_sequences.extend(sequences)
        
        # Calculate metrics
        metrics = self.get_test_metrics(all_predictions, all_labels)
        
        # Print results
        print("\nTest Results:")
        print(f"Position Accuracy: {metrics['position_accuracy']:.4f}")
        print(f"Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
        
        if num_examples > 0:
            print("\nSample Predictions:")
            print("Original\t\tPredicted\tActual")
            print("(first5...last5)\t(first5...last5)\t(first5...last5)")
            print("-" * 70)
            
            indices = np.random.choice(len(all_sequences), min(num_examples, len(all_sequences)), replace=False)
            for idx in indices:
                # Get original sequence start/end
                orig_seq = all_sequences[idx]
                orig_first5 = orig_seq[:5]
                orig_last5 = orig_seq[-5:] if len(orig_seq) > 5 else ""
                
                # Get predicted sequence start/end
                pred_seq = self._indices_to_sequence(all_predictions[idx])
                pred_first5 = pred_seq[:5]
                pred_last5 = pred_seq[-5:] if len(pred_seq) > 5 else ""
                
                # Get actual sequence start/end
                actual_seq = self._indices_to_sequence(all_labels[idx])
                actual_first5 = actual_seq[:5]
                actual_last5 = actual_seq[-5:] if len(actual_seq) > 5 else ""
                
                print(f"{orig_first5}...{orig_last5}\t{pred_first5}...{pred_last5}\t{actual_first5}...{actual_last5}")
        
        return metrics
    
    def get_test_metrics(self, predictions, labels):
        """Calculate reverse complement specific metrics with sequence alignment"""
        position_matches = 0
        total_positions = 0
        sequence_matches = 0
        total_sequences = len(predictions)
        
        # Calculate metrics while preserving sequence alignment
        for pred_seq, label_seq in zip(predictions, labels):
            # Position-wise accuracy
            matches = (pred_seq == label_seq).sum()
            position_matches += matches
            total_positions += len(pred_seq)
            
            # Sequence-wise accuracy
            if len(pred_seq) == len(label_seq) and (pred_seq == label_seq).all():
                sequence_matches += 1
        
        return {
            'position_accuracy': float(position_matches) / total_positions if total_positions > 0 else 0.0,
            'sequence_accuracy': float(sequence_matches) / total_sequences if total_sequences > 0 else 0.0
        }

    def _indices_to_sequence(self, indices):
        """Convert numerical indices back to nucleotide sequence"""
        return ''.join(IDX_TO_NUCLEOTIDE.get(idx, 'N') for idx in indices)

    def show_training_examples(self, dataset, num_examples=10):
        """Show examples of input sequences and their reverse complements"""
        print("\nTraining Examples:")
        print("Original Sequence\tReverse Complement")
        print("-" * 60)
        
        # Get random indices
        indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
        
        for idx in indices:
            example = dataset[idx]
            # Get original sequence
            orig_seq = self.processing_class.decode(example['input_ids'], skip_special_tokens=True)
            # Get reverse complement from labels
            rev_comp = ''.join(IDX_TO_NUCLEOTIDE[idx] for idx in example['rev_comp_labels'])
            print(f"{orig_seq}\t{rev_comp}")
        print()

    def train(self, *args, **kwargs):
        """Override train method to show examples before training"""
        if self.train_dataset is not None:
            self.show_training_examples(self.train_dataset)
        return super().train(*args, **kwargs)

# Register implementation
DNAModelRegistry.register(
    "reverse_complement",
    ReverseComplementHead,
    ReverseComplementDataGenerator,
    ReverseComplementTrainer
)