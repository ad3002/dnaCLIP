from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import DataCollatorWithPadding, TrainingArguments
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple

@dataclass
class PromoterDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Collect promoter labels if available
        labels = [f["labels"] for f in features if "labels" in f]
        
        # Proceed with default collation
        batch = super().__call__(features)
        
        # Add labels if they were collected
        if labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batch

class PromoterDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.data_collator = None
    
    def generate_features(self, sequence):
        # For promoter prediction, we don't need additional features
        # as we're doing binary classification based on sequence alone
        return None
    
    def prepare_dataset(self, dataset, tokenizer):
        # Create custom data collator
        self.data_collator = PromoterDataCollator(tokenizer=tokenizer)

        def preprocess_function(examples):
            # Tokenize sequences with fixed padding
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None
            )
            
            # Add labels (promoter presence)
            tokenized['labels'] = examples['promoter_presence']
            
            return tokenized
            
        return dataset.map(preprocess_function, batched=True)

class PromoterHead(BaseHead):
    def __init__(self, input_dim=768, n_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, sequence_features, **kwargs):
        logits = self.classifier(sequence_features)
        return logits  # Return raw logits, don't apply softmax
    
    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def test(
        self,
        outputs: torch.Tensor,
        sequences: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, float], Optional[List[Dict[str, Union[str, float]]]]]:
        """
        Test method for promoter prediction.
        Args:
            outputs: Model outputs (logits)
            sequences: Original sequences (optional)
            labels: Ground truth labels (optional)
        Returns:
            Tuple of (metrics_dict, predictions_list)
        """
        predictions = outputs.argmax(dim=1)
        metrics = {}
        
        if labels is not None:
            accuracy = (predictions == labels).float().mean().item()
            true_pos = ((predictions == 1) & (labels == 1)).sum().item()
            false_pos = ((predictions == 1) & (labels == 0)).sum().item()
            false_neg = ((predictions == 0) & (labels == 1)).sum().item()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Create detailed predictions if sequences are provided
        detailed_predictions = None
        if sequences is not None:
            detailed_predictions = []
            for i, seq in enumerate(sequences):
                pred_dict = {
                    'sequence': seq,
                    'prediction': predictions[i].item(),
                }
                if labels is not None:
                    pred_dict['actual'] = labels[i].item()
                detailed_predictions.append(pred_dict)

        return metrics, detailed_predictions

def compute_metrics(predictions, labels):
    predictions = predictions.argmax(dim=1)
    return {
        'accuracy': (predictions == labels).float().mean().item()
    }

class PromoterTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get('args', None)
        if training_args:
            training_args.label_names = ["labels"]
            
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
            
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_default_args(output_dir: str) -> TrainingArguments:
        args = BaseTrainer.get_default_args(output_dir)
        args.label_names = ["labels"]
        args.metric_for_best_model = "eval_accuracy"
        args.greater_is_better = True
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.logging_steps = 100
        return args

    def _prepare_inputs(self, inputs):
        """Prepare inputs before compute_loss is called"""
        prepared = super()._prepare_inputs(inputs)
        if isinstance(inputs, dict) and "labels" in inputs:
            prepared["labels"] = inputs["labels"]
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with support for additional kwargs from Trainer
        """
        if not isinstance(inputs, dict):
            inputs = {k: v for k, v in inputs.items()}
            
        labels = inputs.get("labels")
        if labels is None:
            raise KeyError(f"No labels found in inputs. Available keys: {inputs.keys()}")
            
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        loss = model.head.compute_loss(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step"""
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            loss = model.head.compute_loss(outputs, labels)
            
        return (loss.detach(), outputs.detach(), labels)

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Handle both torch.Tensor and numpy.ndarray
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach()
            # Ensure predictions has shape (batch_size, num_classes)
            if predictions.ndim == 1:
                predictions = predictions.unsqueeze(-1)
            predictions = predictions.argmax(dim=1).numpy()
        else:
            # Ensure predictions has shape (batch_size, num_classes)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            predictions = predictions.argmax(axis=1)
            
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
            
        accuracy = (predictions == labels).mean()
        
        # Calculate precision, recall, and F1 for positive class
        true_pos = ((predictions == 1) & (labels == 1)).sum()
        false_pos = ((predictions == 1) & (labels == 0)).sum()
        false_neg = ((predictions == 0) & (labels == 1)).sum()
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def test_promoter_implementation(model, test_dataset, tokenizer, num_examples=10):
    """Standalone test function for promoter prediction implementation"""
    if test_dataset is None or len(test_dataset) == 0:
        raise ValueError("Test dataset is empty or None")

    model = model.eval()
    all_predictions = []
    all_labels = []
    all_sequences = []
    batch_size = 32

    # Ensure test_dataset has the required format
    if not hasattr(test_dataset, '__len__'):
        raise ValueError("Test dataset must be a sequence-like object")

    # Create data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer)

    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i+batch_size]
        
        # Extract features from batch
        features = [{
            'input_ids': batch['input_ids'][j],
            'attention_mask': batch['attention_mask'][j],
            'labels': batch['labels'][j] if 'labels' in batch else batch['promoter_presence'][j]
        } for j in range(len(batch['input_ids']))]
        
        # Use data collator to pad the batch
        padded_batch = data_collator(features)
        
        # Ensure tensors are on the correct device
        device = next(model.parameters()).device
        inputs = {
            'input_ids': padded_batch['input_ids'].to(device),
            'attention_mask': padded_batch['attention_mask'].to(device)
        }
        
        with torch.no_grad():
            # Get raw logits
            logits = model(**inputs)
            if isinstance(logits, torch.Tensor):
                logits = logits.cpu()
            
            # Check and fix logits shape
            if logits.ndim == 1:
                logits = logits.view(-1, 2)
            
            # Get class predictions
            predictions = F.softmax(logits, dim=1).argmax(dim=1).numpy()
        
        labels = [f['labels'] for f in features]
        sequences = batch.get('original_sequence', 
                            [tokenizer.decode(seq, skip_special_tokens=True) 
                             for seq in batch['input_ids']])
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_sequences.extend(sequences)

    # Convert to numpy arrays and reshape if needed
    all_predictions = np.array(all_predictions)
    if all_predictions.ndim == 1:
        all_predictions = all_predictions.reshape(-1)
    all_labels = np.array(all_labels)
    
    # Calculate metrics using the trainer's static method
    metrics = PromoterTrainer.compute_metrics((logits.numpy(), all_labels))
    
    # Print results
    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    if num_examples > 0:
        print("\nSample Predictions:")
        print("Sequence\t\tPredicted\tActual")
        print("-" * 70)
        
        indices = np.random.choice(len(all_predictions), min(num_examples, len(all_predictions)), replace=False)
        for idx in indices:
            seq = all_sequences[idx][:20]
            pred = all_predictions[idx]
            actual = all_labels[idx]
            print(f"{seq}...\t{pred}\t\t{actual}")
    
    return metrics

# Register implementation with test function
DNAModelRegistry.register(
    "promoter",
    PromoterHead,
    PromoterDataGenerator,
    PromoterTrainer,
    test_promoter_implementation
)