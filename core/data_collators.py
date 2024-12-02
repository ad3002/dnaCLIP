from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from typing import Dict, List, Union, Optional, Any
import torch

@dataclass
class BaseDNADataCollator(DataCollatorWithPadding):
    label_name: str
    label_type: torch.dtype = torch.float32  # Add support for different label types
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract values and sequences
        label_values = []
        original_sequences = []
        filtered_features = []
        
        for f in features:
            feature_copy = dict(f)
            if self.label_name in feature_copy:
                value = feature_copy.pop(self.label_name)  # Remove to avoid duplication
                label_values.append(
                    value if isinstance(value, torch.Tensor) 
                    else torch.tensor(value, dtype=self.label_type)
                )
            if "original_sequence" in feature_copy:
                original_sequences.append(feature_copy.pop("original_sequence"))
            filtered_features.append(feature_copy)
        
        # Proceed with default collation
        batch = super().__call__(filtered_features)
        
        # Add collected data
        if label_values:
            if isinstance(label_values[0], torch.Tensor) and label_values[0].dim() > 0:
                batch["labels"] = torch.stack(label_values)
            else:
                batch["labels"] = torch.tensor(label_values, dtype=self.label_type)
        if original_sequences:
            batch["original_sequence"] = original_sequences
            
        return batch

# Add specialized collators
@dataclass
class ClassificationDataCollator(BaseDNADataCollator):
    def __init__(self, tokenizer: Any, label_name: str = "labels"):
        super().__init__(tokenizer=tokenizer, label_name=label_name, label_type=torch.long)

@dataclass
class RegressionDataCollator(BaseDNADataCollator):
    def __init__(self, tokenizer: Any, label_name: str = "labels"):
        super().__init__(tokenizer=tokenizer, label_name=label_name, label_type=torch.float32)