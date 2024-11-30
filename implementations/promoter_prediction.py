from ..core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from ..core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch

class PromoterDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
    
    def generate_features(self, sequence):
        # For promoter prediction, we don't need additional features
        # as we're doing binary classification based on sequence alone
        return None
    
    def prepare_dataset(self, dataset, tokenizer):
        def preprocess_function(examples):
            # Tokenize sequences with fixed padding
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None
            )
            
            # Add labels (promoter presence)
            tokenized['labels'] = examples['promoter_presence']
            
            return tokenized
            
        return dataset.map(preprocess_function, batched=True)

class PromoterHead(BaseHead):
    def __init__(self, input_dim=768, n_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, n_classes)
        )
    
    def forward(self, sequence_features, **kwargs):
        return self.classifier(sequence_features)
    
    def compute_loss(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

def compute_metrics(predictions, labels):
    predictions = predictions.argmax(dim=1)
    return {
        'accuracy': (predictions == labels).float().mean().item()
    }

class PromoterTrainer(BaseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        loss = model.head.compute_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(dim=1)
        return {
            'accuracy': (predictions == labels).float().mean().item()
        }

# Register implementation after all classes are defined
DNAModelRegistry.register(
    "promoter",
    PromoterHead,
    PromoterDataGenerator,
    PromoterTrainer
)