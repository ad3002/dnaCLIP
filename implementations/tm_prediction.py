# filepath: /Users/akomissarov/Dropbox2/Dropbox/workspace/misha/dnaCLIP/implementations/tm_prediction.py
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
class TmDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        tm_values = [f["melting_temp"] for f in features if "melting_temp" in f]
        batch = super().__call__(features)
        if tm_values:
            batch["labels"] = torch.tensor(tm_values, dtype=torch.float32)
        return batch

class TmDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.data_collator = None
        
        # Nearest-neighbor parameters (SantaLucia & Hicks, 2004)
        self.nn_params = {
            'AA/TT': (-7.9, -22.2),
            'AT/TA': (-7.2, -20.4),
            'TA/AT': (-7.2, -21.3),
            'CA/GT': (-8.5, -22.7),
            'GT/CA': (-8.4, -22.4),
            'CT/GA': (-7.8, -21.0),
            'GA/CT': (-8.2, -22.2),
            'CG/GC': (-10.6, -27.2),
            'GC/CG': (-9.8, -24.4),
            'GG/CC': (-8.0, -19.9),
        }
        
        # Salt concentration (mM) and other parameters
        self.Na = 50  # mM Na+
        self.R = 1.987  # Gas constant in cal/K·mol
        
    def generate_features(self, sequence):
        """Implementation of abstract method to generate features"""
        if not sequence:
            return 0.0
        return self.calculate_tm(sequence)
    
    def calculate_tm(self, sequence):
        """Calculate melting temperature using nearest-neighbor method"""
        sequence = sequence.upper()
        if len(sequence) < 2:
            return 0.0
            
        # Initialize thermodynamic parameters
        dH = 0  # Enthalpy
        dS = 0  # Entropy
        
        # Calculate nearest-neighbor contributions
        for i in range(len(sequence) - 1):
            pair = sequence[i:i+2]
            complement = ''.join(['T' if b == 'A' else 'A' if b == 'T' else 'G' if b == 'C' else 'C' for b in pair[::-1]])
            key = f"{pair}/{complement}"
            
            # Use the parameters if available, otherwise use average values
            if key in self.nn_params:
                h, s = self.nn_params[key]
                dH += h
                dS += s
            else:
                # Average values for non-standard pairs
                dH += -8.0
                dS += -22.0
        
        # Terminal AT penalty
        if sequence[0] in 'AT':
            dH += 2.3
            dS += 4.1
        if sequence[-1] in 'AT':
            dH += 2.3
            dS += 4.1
        
        # Salt correction
        salt_correction = 0.368 * (len(sequence) - 1) * np.log(self.Na/1000)
        dS += salt_correction
        
        # Calculate Tm
        tm = (1000 * dH) / (dS + self.R * np.log(1/4)) - 273.15
        
        # Normalize to [0,1] range, assuming Tm typically falls between 0°C and 120°C
        normalized_tm = (tm - 0) / (120 - 0)
        return max(0.0, min(1.0, normalized_tm))
    
    def prepare_dataset(self, dataset, tokenizer):
        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            tm_values = [
                self.calculate_tm(seq) for seq in examples["sequence"]
            ]
            
            tokenized["melting_temp"] = tm_values
            tokenized["original_sequence"] = examples["sequence"]
            
            return tokenized
                
        self.data_collator = TmDataCollator(tokenizer=tokenizer)
        
        processed_dataset = {}
        for split in dataset.keys():
            processed_dataset[split] = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Processing {split} split"
            )
                
        return processed_dataset

class TmHead(BaseHead):
    def __init__(self, input_dim=1024):
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
        """Test method specifically for Tm prediction"""
        with torch.no_grad():
            return self.forward(sequence_features, **kwargs)

class TmTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get('args', None)
        if training_args:
            training_args.label_names = ["melting_temp"]
        
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
            
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_default_args(output_dir: str) -> TrainingArguments:
        args = BaseTrainer.get_default_args(output_dir)
        args.label_names = ["melting_temp"]
        args.metric_for_best_model = "eval_mae"
        args.greater_is_better = False
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.logging_steps = 100
        return args

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
            
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        return {
            'mse': mse,
            'mae': mae
        }

    def _prepare_inputs(self, inputs):
        """Prepare inputs before compute_loss is called"""
        prepared = super()._prepare_inputs(inputs)
        if isinstance(inputs, dict) and "melting_temp" in inputs:
            prepared["labels"] = inputs["melting_temp"]
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Convert inputs to dict if it's not already
        if not isinstance(inputs, dict):
            inputs = {k: v for k, v in inputs.items()}
            
        # Get the labels
        labels = inputs.get("labels", None)
        if labels is None:
            labels = inputs.get("melting_temp", None)
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
        tm_content = inputs["melting_temp"] if "melting_temp" in inputs else inputs["labels"]
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            loss = model.head.compute_loss(outputs, tm_content)
            
        return (loss.detach(), outputs.detach(), tm_content)

def test_tm_implementation(model, test_dataset, tokenizer, num_examples=10):
    """Standalone test function for Tm implementation"""
    if test_dataset is None or len(test_dataset) == 0:
        raise ValueError("Test dataset is empty or None")

    model = model.eval()
    all_predictions = []
    all_labels = []
    all_sequences = []
    batch_size = 32

    data_collator = DataCollatorWithPadding(tokenizer)

    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i+batch_size]
        
        features = [{
            'input_ids': batch['input_ids'][j],
            'attention_mask': batch['attention_mask'][j]
        } for j in range(len(batch['input_ids']))]
        
        padded_batch = data_collator(features)
        
        inputs = {
            'input_ids': padded_batch['input_ids'],
            'attention_mask': padded_batch['attention_mask']
        }
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            predictions = model(**inputs).cpu().numpy().squeeze()
        
        labels = batch['melting_temp']
        sequences = batch.get('original_sequence', 
                            [tokenizer.decode(seq, skip_special_tokens=True) 
                             for seq in batch['input_ids']])
        
        all_predictions.extend(predictions if isinstance(predictions, list) else predictions.tolist())
        all_labels.extend(labels)
        all_sequences.extend(sequences)
    
    metrics = TmTrainer.compute_metrics((np.array(all_predictions), np.array(all_labels)))
    
    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    if num_examples > 0:
        print("\nSample Predictions:")
        print("Sequence\t\tPredicted Tm\tActual Tm\tDiff")
        print("-" * 70)
        
        indices = np.random.choice(len(all_predictions), min(num_examples, len(all_predictions)), replace=False)
        for idx in indices:
            seq = all_sequences[idx]
            pred = all_predictions[idx]
            actual = all_labels[idx]
            print(f"{seq[:20]}...\t{pred:.3f}\t\t{actual:.3f}\t\t{abs(pred-actual):.3f}")
    
    return metrics

# Register Tm prediction implementation
DNAModelRegistry.register(
    "tm_prediction",
    TmHead,
    TmDataGenerator,
    TmTrainer,
    test_tm_implementation
)