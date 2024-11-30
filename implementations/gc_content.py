from ..core.base_classes import BaseDataGenerator, BaseHead, BaseTrainer
from ..core.registry import register_dna_model
import torch.nn as nn
import torch.nn.functional as F
import torch

class GCContentDataGenerator(BaseDataGenerator):
    def __init__(self, max_length=128):
        self.max_length = max_length
    
    def generate_features(self, sequence):
        gc_count = sum(1 for base in sequence if base in 'GC')
        return gc_count / len(sequence)
    
    def prepare_dataset(self, dataset, tokenizer):
        def preprocess_function(examples):
            tokenized = tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None
            )
            tokenized['gc_content'] = [
                self.generate_features(seq) for seq in examples["sequence"]
            ]
            return tokenized
        return dataset.map(preprocess_function, batched=True)

@register_dna_model("gc_content")
class GCContentHead(BaseHead):
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

class GCContentTrainer(BaseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        gc_content = inputs.pop("gc_content")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        loss = model.head.compute_loss(outputs, gc_content)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        mse = F.mse_loss(predictions, labels)
        return {
            'mse': mse.item(),
            'correlation': torch.corrcoef(torch.stack([predictions, labels]))[0,1].item()
        }
    
    def evaluate(self, model, eval_dataset):
        model.eval()
        total_mse = 0
        n_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataset:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                mse = F.mse_loss(outputs.squeeze(), batch['gc_content'].float())
                total_mse += mse.item() * len(batch['gc_content'])
                n_samples += len(batch['gc_content'])
        
        return {'mse': total_mse / n_samples}