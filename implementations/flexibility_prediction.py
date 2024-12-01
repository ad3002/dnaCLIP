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
class FlexibilityDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        flex_values = [f["flexibility"] for f in features if "flexibility" in f]
        batch = super().__call__(features)
        if flex_values:
            batch["labels"] = torch.tensor(flex_values, dtype=torch.float32)
        return batch

class FlexibilityDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.data_collator = None
        
        # DNA flexibility parameters (trinucleotide scale)
        # Values from Brukner et al. (1995) and other sources
        self.flexibility_params = {
            # Trinucleotide: (propeller twist, bendability)
            'AAA': (0.26, 0.14), 'AAT': (0.22, 0.16), 'AAG': (0.20, 0.18),
            'AAC': (0.19, 0.17), 'ATA': (0.20, 0.15), 'ATT': (0.17, 0.14),
            'ATG': (0.18, 0.16), 'ATC': (0.16, 0.15), 'AGA': (0.19, 0.17),
            'AGT': (0.17, 0.16), 'AGG': (0.18, 0.18), 'AGC': (0.16, 0.17),
            'ACA': (0.17, 0.16), 'ACT': (0.15, 0.15), 'ACG': (0.16, 0.17),
            'ACC': (0.14, 0.16), 'TAA': (0.22, 0.15), 'TAT': (0.19, 0.14),
            'TAG': (0.17, 0.16), 'TAC': (0.16, 0.15), 'TTA': (0.19, 0.14),
            'TTT': (0.26, 0.14), 'TTG': (0.15, 0.16), 'TTC': (0.14, 0.15),
            'TGA': (0.17, 0.16), 'TGT': (0.15, 0.15), 'TGG': (0.16, 0.17),
            'TGC': (0.14, 0.16), 'TCA': (0.16, 0.15), 'TCT': (0.14, 0.14),
            'TCG': (0.14, 0.16), 'TCC': (0.12, 0.15), 'GAA': (0.20, 0.18),
            'GAT': (0.17, 0.16), 'GAG': (0.18, 0.19), 'GAC': (0.16, 0.17),
            'GTA': (0.18, 0.16), 'GTT': (0.15, 0.15), 'GTG': (0.16, 0.17),
            'GTC': (0.14, 0.16), 'GGA': (0.18, 0.18), 'GGT': (0.16, 0.17),
            'GGG': (0.17, 0.19), 'GGC': (0.15, 0.18), 'GCA': (0.16, 0.17),
            'GCT': (0.14, 0.16), 'GCG': (0.15, 0.18), 'GCC': (0.13, 0.17),
            'CAA': (0.19, 0.17), 'CAT': (0.16, 0.15), 'CAG': (0.16, 0.17),
            'CAC': (0.14, 0.16), 'CTA': (0.17, 0.15), 'CTT': (0.14, 0.14),
            'CTG': (0.14, 0.16), 'CTC': (0.12, 0.15), 'CGA': (0.16, 0.17),
            'CGT': (0.14, 0.16), 'CGG': (0.15, 0.18), 'CGC': (0.13, 0.17),
            'CCA': (0.14, 0.16), 'CCT': (0.12, 0.15), 'CCG': (0.13, 0.17),
            'CCC': (0.11, 0.16)
        }
    
    def generate_features(self, sequence):
        if not sequence:
            return [0.0, 0.0]
        return self.calculate_flexibility(sequence)
    
    def calculate_flexibility(self, sequence):
        """Calculate average propeller twist and bendability"""
        sequence = sequence.upper()
        if len(sequence) < 3:
            return [0.0, 0.0]
        
        propeller_sum = 0.0
        bendability_sum = 0.0
        count = 0
        
        # Slide window of 3 nucleotides
        for i in range(len(sequence) - 2):
            trinuc = sequence[i:i+3]
            if trinuc in self.flexibility_params:
                prop, bend = self.flexibility_params[trinuc]
                propeller_sum += prop
                bendability_sum += bend
                count += 1
        
        if count == 0:
            return [0.0, 0.0]
        
        # Return normalized average values
        return [
            propeller_sum / count,
            bendability_sum / count
        ]
    
    def prepare_dataset(self, dataset, tokenizer):
        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            
            flex_values = [
                self.calculate_flexibility(seq) for seq in examples["sequence"]
            ]
            
            # Split flexibility values into two features
            tokenized["flexibility"] = flex_values
            tokenized["original_sequence"] = examples["sequence"]
            
            return tokenized
        
        self.data_collator = FlexibilityDataCollator(tokenizer=tokenizer)
        
        processed_dataset = {}
        for split in dataset.keys():
            processed_dataset[split] = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Processing {split} split"
            )
        
        return processed_dataset

class FlexibilityHead(BaseHead):
    def __init__(self, input_dim=768):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # 2 outputs: propeller twist and bendability
            nn.Sigmoid()
        )
    
    def forward(self, sequence_features, **kwargs):
        return self.regressor(sequence_features)
    
    def compute_loss(self, outputs, targets):
        return F.mse_loss(outputs, targets.float())
    
    def test(self, sequence_features, **kwargs):
        with torch.no_grad():
            return self.forward(sequence_features, **kwargs)

class FlexibilityTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        training_args = kwargs.get('args', None)
        if training_args:
            training_args.label_names = ["flexibility"]
        
        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = self.compute_metrics
            
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_default_args(output_dir: str) -> TrainingArguments:
        args = BaseTrainer.get_default_args(output_dir)
        args.label_names = ["flexibility"]
        args.metric_for_best_model = "eval_mae"
        args.greater_is_better = False
        args.evaluation_strategy = "steps"
        args.eval_steps = 100
        args.logging_steps = 100
        return args

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
            
        # Calculate metrics for both propeller twist and bendability
        mse_prop = np.mean((predictions[:, 0] - labels[:, 0]) ** 2)
        mae_prop = np.mean(np.abs(predictions[:, 0] - labels[:, 0]))
        mse_bend = np.mean((predictions[:, 1] - labels[:, 1]) ** 2)
        mae_bend = np.mean(np.abs(predictions[:, 1] - labels[:, 1]))
        
        return {
            'mse_propeller': mse_prop,
            'mae_propeller': mae_prop,
            'mse_bendability': mse_bend,
            'mae_bendability': mae_bend,
            'mae': (mae_prop + mae_bend) / 2  # Average MAE for model selection
        }

    def _prepare_inputs(self, inputs):
        """Prepare inputs before compute_loss is called"""
        prepared = super()._prepare_inputs(inputs)
        if isinstance(inputs, dict) and "flexibility" in inputs:
            prepared["labels"] = inputs["flexibility"]
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not isinstance(inputs, dict):
            inputs = {k: v for k, v in inputs.items()}
            
        labels = inputs.get("labels", None)
        if labels is None:
            labels = inputs.get("flexibility", None)
        if labels is None:
            raise KeyError(f"No labels found in inputs. Available keys: {inputs.keys()}")
            
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        loss = model.head.compute_loss(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss

    def test_model(self, test_dataset, num_examples=10):
        model = self.model.eval()
        all_predictions = []
        all_labels = []
        all_sequences = []
        
        for i in range(0, len(test_dataset), self.args.per_device_eval_batch_size):
            batch = test_dataset[i:i+self.args.per_device_eval_batch_size]
            inputs = self._prepare_inputs(batch)
            
            with torch.no_grad():
                predictions = model(**inputs).cpu().numpy()
            
            labels = batch['flexibility']
            sequences = [
                self.tokenizer.decode(seq, skip_special_tokens=True) 
                for seq in batch['input_ids']
            ]
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_sequences.extend(sequences)
        
        metrics = self.compute_metrics((np.array(all_predictions), np.array(all_labels)))
        
        print("\nTest Results:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        if num_examples > 0:
            print("\nSample Predictions:")
            print("Sequence\t\tPredicted (Prop, Bend)\tActual (Prop, Bend)\tDiff")
            print("-" * 80)
            
            indices = np.random.choice(len(all_predictions), num_examples, replace=False)
            for idx in indices:
                seq = all_sequences[idx]
                pred = all_predictions[idx]
                actual = all_labels[idx]
                diff = np.abs(pred - actual)
                print(f"{seq[:20]}...\t({pred[0]:.3f}, {pred[1]:.3f})\t\t({actual[0]:.3f}, {actual[1]:.3f})\t\t({diff[0]:.3f}, {diff[1]:.3f})")
        
        return {
            **metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'sequences': all_sequences
        }

    def get_test_metrics(self, predictions, labels):
        """Calculate flexibility-specific metrics"""
        return self.compute_metrics((predictions, labels))

def test_flexibility_implementation(model, test_dataset, tokenizer, num_examples=10):
    """Standalone test function for flexibility prediction"""
    trainer = FlexibilityTrainer(model=model, args=None)
    return trainer.test_model(test_dataset, num_examples)

# Update registration to use the proper trainer and test implementation
DNAModelRegistry.register(
    "flexibility",
    FlexibilityHead,
    FlexibilityDataGenerator,
    FlexibilityTrainer,
    test_flexibility_implementation
)