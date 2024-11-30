from dnaCLIP.core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
import torch.nn as nn
import torch.nn.functional as F
import torch

class GcContentDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
    
    def generate_features(self, sequence):
        gc_count = sum(1 for base in sequence.upper() if base in ['G', 'C'])
        return gc_count / len(sequence) if sequence else 0.0
    
    def prepare_dataset(self, dataset, tokenizer):
        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None
            )
            # Calculate GC content and convert to float32
            tokenized['labels'] = torch.tensor(
                [self.generate_features(seq) for seq in examples["sequence"]],
                dtype=torch.float32
            )
            return tokenized
            
        return dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )

class GcContentHead(BaseHead):
    def __init__(self, input_dim=768):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequence_features, **kwargs):
        return self.regressor(sequence_features)
    
    def compute_loss(self, outputs, targets):
        return F.mse_loss(outputs.squeeze(), targets)

class GcContentTrainer(BaseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')
        if labels is None and hasattr(inputs, 'data'):
            labels = inputs.data.get('labels')
            inputs = inputs.data
            
        if labels is None:
            raise ValueError("No labels found in inputs. Input keys: " + str(inputs.keys()))
            
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        # Ensure labels are float32
        labels = labels.to(dtype=torch.float32)
        loss = model.head.compute_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        mse = F.mse_loss(torch.tensor(predictions), torch.tensor(labels))
        return {'mse': mse.item()}

# Register implementation
DNAModelRegistry.register(
    "gc_content",
    GcContentHead,
    GcContentDataGenerator,
    GcContentTrainer
)