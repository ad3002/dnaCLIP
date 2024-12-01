from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import TrainingArguments
from typing import Dict, List, Union, Optional, Tuple

class PromoterDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
    
    def generate_features(self, sequence):
        # For promoter prediction, we don't need additional features
        # as we're doing binary classification based on sequence alone
        return None
    
    def prepare_dataset(self, dataset, tokenizer):
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
            nn.Linear(input_dim, n_classes)
        )
    
    def forward(self, sequence_features, **kwargs):
        return self.classifier(sequence_features)
    
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

    def compute_loss(self, model, inputs, return_outputs=False):
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
        predictions = predictions.argmax(dim=1)
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
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

def test_promoter_implementation(model, dataset, tokenizer, num_examples=10):
    """Standalone test function for promoter prediction implementation"""
    model = model.eval()
    all_predictions = []
    all_labels = []
    all_sequences = []
    
    for i in range(0, len(dataset), 32):  # Using default batch size of 32
        batch = dataset[i:i+32]
        inputs = tokenizer(
            batch['sequence'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.argmax(dim=1).numpy()
        
        labels = batch['promoter_presence']
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_sequences.extend(batch['sequence'])
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate basic metrics
    accuracy = (all_predictions == all_labels).mean()
    true_pos = ((all_predictions == 1) & (all_labels == 1)).sum()
    false_pos = ((all_predictions == 1) & (all_labels == 0)).sum()
    false_neg = ((all_predictions == 0) & (all_labels == 1)).sum()
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\nTest Results:")
    print(f"ACCURACY: {accuracy:.4f}")
    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    if num_examples > 0:
        print("\nSample Predictions:")
        print("Sequence\t\tPredicted\tActual")
        print("-" * 70)
        
        indices = np.random.choice(len(all_predictions), num_examples, replace=False)
        for idx in indices:
            seq = all_sequences[idx][:20]
            pred = all_predictions[idx]
            actual = all_labels[idx]
            print(f"{seq}...\t{pred}\t\t{actual}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'sequences': all_sequences
    }

# Register implementation with test function
DNAModelRegistry.register(
    "promoter",
    PromoterHead,
    PromoterDataGenerator,
    PromoterTrainer,
    test_promoter_implementation
)