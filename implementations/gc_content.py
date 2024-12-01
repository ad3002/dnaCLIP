from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np  # Add this import
from transformers import DataCollatorWithPadding, TrainingArguments  # Added TrainingArguments import
from dataclasses import dataclass
from typing import Dict, List, Union

@dataclass
class GCDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Collect gc_content if available
        gc_values = [f["gc_content"] for f in features if "gc_content" in f]
        
        # Proceed with default collation
        batch = super().__call__(features)
        
        # Add labels if gc_values were collected
        if (gc_values):
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
            # Tokenize sequences without padding
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Let the data collator handle padding
                return_tensors=None
            )
            
            # Calculate and add GC content
            gc_contents = [
                self.generate_features(seq) for seq in examples["sequence"]
            ]
            
            # Include gc_content and original sequence in the tokenized output
            tokenized["gc_content"] = gc_contents
            tokenized["original_sequence"] = examples["sequence"]  # Preserve original sequences
            
            return tokenized
                
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
    
    def test(self, sequence_features, **kwargs):
        """Test method specifically for GC content prediction"""
        with torch.no_grad():
            return self.forward(sequence_features, **kwargs)

class GcContentTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get('args', None)
        if training_args:
            # Ensure label_names is set
            training_args.label_names = ["gc_content"]
            
        # Set compute_metrics before parent initialization
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
            
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_default_args(output_dir: str) -> TrainingArguments:
        args = BaseTrainer.get_default_args(output_dir)
        args.label_names = ["gc_content"]
        args.metric_for_best_model = "eval_correlation"  # Add this
        args.greater_is_better = True  # Higher correlation is better
        args.evaluation_strategy = "steps"  # Change to more frequent evaluation
        args.eval_steps = 100  # Evaluate every 100 steps
        args.logging_steps = 100  # Log every 100 steps
        return args

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        # Calculate metrics
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        correlation = np.corrcoef(predictions.flatten(), labels.flatten())[0,1]
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }

    def _prepare_inputs(self, inputs):
        """Prepare inputs before compute_loss is called"""
        prepared = super()._prepare_inputs(inputs)
        if isinstance(inputs, dict) and "gc_content" in inputs:
            prepared["labels"] = inputs["gc_content"]
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Convert inputs to dict if it's not already
        if not isinstance(inputs, dict):
            inputs = {k: v for k, v in inputs.items()}
            
        # Get the labels
        labels = inputs.get("labels", None)
        if labels is None:
            labels = inputs.get("gc_content", None)
        if labels is None:
            raise KeyError(f"No labels found in inputs. Available keys: {inputs.keys()}")
            
        # Get model outputs
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Compute loss
        loss = model.head.compute_loss(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss

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

    def test_model(self, dataset, num_examples=10):
        """Test GC content prediction model with detailed analysis"""
        model = self.model.eval()
        tokenizer = self.tokenizer
        all_predictions = []
        all_labels = []
        all_sequences = []
        
        # Process test set in batches
        for i in range(0, len(dataset), self.args.per_device_eval_batch_size):
            batch = dataset[i:i+self.args.per_device_eval_batch_size]
            inputs = self._prepare_inputs({
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            })
            
            with torch.no_grad():
                predictions = model(**inputs).cpu().numpy().squeeze()
            
            # Get actual GC content
            labels = batch['gc_content']
            sequences = [
                tokenizer.decode(seq, skip_special_tokens=True) 
                for seq in batch['input_ids']
            ]
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_sequences.extend(sequences)
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        mse = np.mean((all_predictions - all_labels) ** 2)
        mae = np.mean(np.abs(all_predictions - all_labels))
        correlation = np.corrcoef(all_predictions, all_labels)[0,1]
        
        print("\nOverall Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Correlation: {correlation:.4f}")
        
        print("\nSample Predictions:")
        print("Sequence\t\tPredicted GC\tActual GC\tDiff")
        print("-" * 70)
        
        # Show sample predictions
        indices = np.random.choice(len(all_predictions), num_examples, replace=False)
        for idx in indices:
            seq = all_sequences[idx]
            pred = all_predictions[idx]
            actual = all_labels[idx]
            print(f"{seq[:20]}...\t{pred:.3f}\t\t{actual:.3f}\t\t{abs(pred-actual):.3f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'predictions': all_predictions,
            'labels': all_labels,
            'sequences': all_sequences
        }
    
    def get_test_metrics(self, predictions, labels):
        """Calculate GC content specific metrics"""
        predictions = np.array(predictions).squeeze()
        labels = np.array(labels)
        
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        correlation = np.corrcoef(predictions.flatten(), labels.flatten())[0,1]
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }
    
    def test_model(self, test_dataset, num_examples=10):
        """Test GC content prediction model"""
        model = self.model.eval()
        all_predictions = []
        all_labels = []
        all_sequences = []
        
        for i in range(0, len(test_dataset), self.args.per_device_eval_batch_size):
            batch = test_dataset[i:i+self.args.per_device_eval_batch_size]
            inputs = self._prepare_inputs(batch)
            
            with torch.no_grad():
                predictions = model(**inputs).cpu().numpy()
            
            labels = batch['gc_content']
            sequences = [
                self.tokenizer.decode(seq, skip_special_tokens=True) 
                for seq in batch['input_ids']
            ]
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_sequences.extend(sequences)
        
        # Calculate metrics
        metrics = self.get_test_metrics(all_predictions, all_labels)
        
        # Print detailed results
        print("\nTest Results:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Show sample predictions
        if num_examples > 0:
            print("\nSample Predictions:")
            print("Sequence\t\tPredicted GC\tActual GC\tDiff")
            print("-" * 70)
            
            indices = np.random.choice(len(all_predictions), num_examples, replace=False)
            for idx in indices:
                seq = all_sequences[idx]
                pred = all_predictions[idx]
                actual = all_labels[idx]
                print(f"{seq[:20]}...\t{pred:.3f}\t\t{actual:.3f}\t\t{abs(pred-actual)::.3f}")
        
        return {
            **metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'sequences': all_sequences
        }

def test_gc_implementation(model, test_dataset, tokenizer, num_examples=10):
    """Standalone test function for GC content implementation"""
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
            'attention_mask': batch['attention_mask'][j]
        } for j in range(len(batch['input_ids']))]
        
        # Use data collator to pad the batch
        padded_batch = data_collator(features)
        
        # Ensure tensors are on the correct device
        inputs = {
            'input_ids': padded_batch['input_ids'],
            'attention_mask': padded_batch['attention_mask']
        }
        
        with torch.no_grad():
            predictions = model(**inputs).cpu().numpy().squeeze()
        
        # Get labels directly from dataset
        labels = batch['gc_content']
        sequences = batch.get('original_sequence', 
                            [tokenizer.decode(seq, skip_special_tokens=True) 
                             for seq in batch['input_ids']])
        
        all_predictions.extend(predictions if isinstance(predictions, list) else predictions.tolist())
        all_labels.extend(labels)
        all_sequences.extend(sequences)
    
    # Calculate metrics using the trainer's static method
    metrics = GcContentTrainer.compute_metrics((np.array(all_predictions), np.array(all_labels)))
    
    # Print results
    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    if num_examples > 0:
        print("\nSample Predictions:")
        print("Sequence\t\tPredicted GC\tActual GC\tDiff")
        print("-" * 70)
        
        indices = np.random.choice(len(all_predictions), min(num_examples, len(all_predictions)), replace=False)
        for idx in indices:
            seq = all_sequences[idx]
            pred = all_predictions[idx]
            actual = all_labels[idx]
            print(f"{seq[:20]}...\t{pred:.3f}\t\t{actual:.3f}\t\t{abs(pred-actual):.3f}")
    
    return metrics

# Register implementation with test function
DNAModelRegistry.register(
    "gc_content",
    GcContentHead,
    GcContentDataGenerator,
    GcContentTrainer,
    test_gc_implementation
)