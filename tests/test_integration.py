# filepath: /Users/akomissarov/Dropbox2/Dropbox/workspace/misha/dnaCLIP/tests/test_integration.py
import pytest
import torch
from dnaCLIP.core.base_classes import BaseDNAModel
from dnaCLIP.implementations.gc_content import GcContentHead, GcContentDataGenerator
from transformers import AutoModel, AutoTokenizer

class TestGCContentIntegration:
    @pytest.fixture(scope="class")
    def model_components(self):
        tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t-multi")
        backbone = AutoModel.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t-multi")
        head = GcContentHead()
        data_generator = GcContentDataGenerator()
        return backbone, head, data_generator, tokenizer
    
    def test_model_forward_pass(self, model_components):
        backbone, head, data_generator, tokenizer = model_components
        model = BaseDNAModel(backbone, head, data_generator)
        
        # Test sequence
        sequence = "ATGCATGC"
        inputs = tokenizer(sequence, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape[-1] == 1  # Single output for GC content
        assert 0 <= outputs.item() <= 1  # Output should be between 0 and 1