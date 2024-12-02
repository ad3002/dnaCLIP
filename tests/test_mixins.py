
import pytest
import numpy as np
from dnaCLIP.core.mixins import NucleotideFeaturesMixin, TriNucleotideFeaturesMixin, MetricsCalculationMixin

class TestNucleotideFeaturesMixin:
    def setup_method(self):
        self.mixin = NucleotideFeaturesMixin()
    
    def test_count_nucleotides(self):
        sequence = "ATGCATGC"
        counts = self.mixin.count_nucleotides(sequence)
        assert counts == {'A': 2, 'T': 2, 'G': 2, 'C': 2}
        
    def test_calculate_gc_content(self):
        sequence = "ATGCATGC"
        gc_content = self.mixin.calculate_gc_content(sequence)
        assert gc_content == 0.5
        
    def test_empty_sequence(self):
        assert self.mixin.calculate_gc_content("") == 0.0

class TestMetricsCalculationMixin:
    def setup_method(self):
        self.mixin = MetricsCalculationMixin()
    
    def test_regression_metrics(self):
        predictions = np.array([0.1, 0.2, 0.3])
        labels = np.array([0.15, 0.25, 0.35])
        metrics = self.mixin.calculate_regression_metrics(predictions, labels)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'correlation' in metrics
        assert metrics['correlation'] > 0.9  # Should be highly correlated
    
    def test_classification_metrics(self):
        predictions = np.array([1, 0, 1, 1])
        labels = np.array([1, 0, 0, 1])
        metrics = self.mixin.calculate_classification_metrics(predictions, labels)
        
        assert metrics['accuracy'] == 0.75
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics