
from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.mixins import MetricsCalculationMixin
from dnaCLIP.core.data_collators import ClassificationDataCollator
from dnaCLIP.core.registry import DNAModelRegistry
from dnaCLIP.core.genome_sampler import GenomeSampler
import torch.nn as nn
import torch.nn.functional as F
import torch

class TaxonomyDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=512):
        super().__init__()
        self.max_length = max_length
        self.labels = {'prokaryote': 0, 'archaea': 1}
        
    def prepare_dataset(self, dataset, tokenizer):
        self.data_collator = ClassificationDataCollator(
            tokenizer=tokenizer,
            label_name="taxonomy"
        )
        
        # Convert labels to integers
        dataset = dataset.map(
            lambda x: {'labels': self.labels[x['taxonomy']]}
        )
        
        return super().prepare_dataset(dataset, tokenizer)

class TaxonomyHead(BaseHead):
    def __init__(self, input_dim=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # 2 classes: prokaryote/archaea
        )
    
    def forward(self, sequence_features, **kwargs):
        return self.classifier(sequence_features)
    
    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

class TaxonomyTrainer(BaseTrainer, MetricsCalculationMixin):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('compute_metrics', self.compute_metrics)
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(-1)
        return MetricsCalculationMixin.calculate_classification_metrics(
            preds, 
            labels
        )

def test_taxonomy_implementation(model, test_dataset, tokenizer, num_examples=10):
    trainer = TaxonomyTrainer(
        model=model,
        args=None,
        tokenizer=tokenizer
    )
    return trainer.test_model(test_dataset, num_examples)

# Register implementation
DNAModelRegistry.register(
    "taxonomy_classification",
    TaxonomyHead,
    TaxonomyDataGenerator,
    TaxonomyTrainer,
    test_taxonomy_implementation
)