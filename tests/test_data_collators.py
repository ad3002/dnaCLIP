# filepath: /Users/akomissarov/Dropbox2/Dropbox/workspace/misha/dnaCLIP/tests/test_data_collators.py
import pytest
import torch
from dnaCLIP.core.data_collators import BaseDNADataCollator, RegressionDataCollator, ClassificationDataCollator

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
    
    def pad(self, *args, **kwargs):
        return {'input_ids': torch.zeros(1, 10)}

class TestDataCollators:
    def setup_method(self):
        self.tokenizer = MockTokenizer()
        
    def test_regression_collator(self):
        collator = RegressionDataCollator(self.tokenizer, label_name="value")
        features = [
            {"input_ids": [1, 2], "value": 0.5},
            {"input_ids": [1, 2, 3], "value": 0.7}
        ]
        
        batch = collator(features)
        assert "labels" in batch
        assert isinstance(batch["labels"], torch.Tensor)
        assert batch["labels"].dtype == torch.float32
        
    def test_classification_collator(self):
        collator = ClassificationDataCollator(self.tokenizer, label_name="label")
        features = [
            {"input_ids": [1, 2], "label": 1},
            {"input_ids": [1, 2, 3], "label": 0}
        ]
        
        batch = collator(features)
        assert "labels" in batch
        assert isinstance(batch["labels"], torch.Tensor)
        assert batch["labels"].dtype == torch.long