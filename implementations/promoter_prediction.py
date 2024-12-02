from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.mixins import MetricsCalculationMixin
from dnaCLIP.core.data_collators import ClassificationDataCollator
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch

class PromoterDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        super().__init__()
        self.max_length = max_length
        self.data_collator = None
    
    def generate_features(self, sequence):
        return None  # No additional features needed for promoter prediction
    
    def prepare_dataset(self, dataset, tokenizer):
        self.data_collator = ClassificationDataCollator(
            tokenizer=tokenizer,
            label_name="labels"
        )

        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            tokenized['labels'] = examples['promoter_presence']
            tokenized['original_sequence'] = examples['sequence']
            return tokenized
            
        processed_dataset = {}
        for split in dataset.keys():
            processed_dataset[split] = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Processing {split} split"
            )
                
        return processed_dataset

class PromoterHead(BaseHead):
    def __init__(self, input_dim=768, n_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, sequence_features, **kwargs):
        return self.classifier(sequence_features)
    
    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def test(self, sequence_features, **kwargs):
        with torch.no_grad():
            return self.forward(sequence_features, **kwargs)

class PromoterTrainer(BaseTrainer, MetricsCalculationMixin):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('compute_metrics', self.compute_metrics)
        super().__init__(*args, **kwargs)

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            
        predictions = predictions.argmax(axis=1) if predictions.ndim > 1 else predictions
        return MetricsCalculationMixin.calculate_classification_metrics(predictions, labels)

    def get_test_metrics(self, predictions, labels):
        return self.compute_metrics((predictions, labels))

def test_promoter_implementation(model, test_dataset, tokenizer, num_examples=10):
    trainer = PromoterTrainer(
        model=model,
        args=None,
        tokenizer=tokenizer
    )
    return trainer.test_model(test_dataset, num_examples)

# Register implementation
DNAModelRegistry.register(
    "promoter",
    PromoterHead,
    PromoterDataGenerator,
    PromoterTrainer,
    test_promoter_implementation
)