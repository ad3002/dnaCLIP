from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.mixins import NucleotideFeaturesMixin, MetricsCalculationMixin
from dnaCLIP.core.data_collators import RegressionDataCollator
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class GcContentDataGenerator(BaseDataGenerator, NucleotideFeaturesMixin):
    def __init__(self, max_length=128):
        super().__init__()
        self.max_length = max_length
        self.data_collator = None
    
    def generate_features(self, sequence):
        return self.calculate_gc_content(sequence)
    
    def prepare_dataset(self, dataset, tokenizer):
        self.data_collator = RegressionDataCollator(
            tokenizer=tokenizer,
            label_name="gc_content"
        )
        
        return super().prepare_dataset(dataset, tokenizer)

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
        with torch.no_grad():
            return self.forward(sequence_features, **kwargs)

class GcContentTrainer(BaseTrainer, MetricsCalculationMixin):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('compute_metrics', self.compute_metrics)
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return MetricsCalculationMixin.calculate_regression_metrics(
            predictions.squeeze(), 
            labels.squeeze()
        )

    def get_test_metrics(self, predictions, labels):
        return self.compute_metrics((predictions, labels))

def test_gc_implementation(model, test_dataset, tokenizer, num_examples=10):
    trainer = GcContentTrainer(
        model=model,
        args=None,
        tokenizer=tokenizer
    )
    return trainer.test_model(test_dataset, num_examples)

# Register implementation
DNAModelRegistry.register(
    "gc_content",
    GcContentHead,
    GcContentDataGenerator,
    GcContentTrainer,
    test_gc_implementation
)