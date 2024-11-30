from abc import ABC, abstractmethod
import torch.nn as nn
from transformers import Trainer, TrainingArguments

class BaseDataGenerator(ABC):
    @abstractmethod
    def generate_features(self, sequence):
        """Generate features from a DNA sequence"""
        pass
    
    @abstractmethod
    def prepare_dataset(self, dataset, tokenizer):
        """Prepare dataset for training"""
        pass

class BaseHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, sequence_features, **kwargs):
        """Forward pass for the head"""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs, targets):
        """Compute loss for the head"""
        pass

class BaseDNAModel(nn.Module):
    def __init__(self, backbone, head, data_generator):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.data_generator = data_generator
    
    def forward(self, input_ids, attention_mask, **kwargs):
        sequence_features = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).hidden_states[-1][:, 0]  # Using [CLS] token
        
        return self.head(sequence_features, **kwargs)

class BaseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation"""
        pass
    
    @staticmethod
    def get_default_args(output_dir: str) -> TrainingArguments:
        """Get default training arguments"""
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            lr_scheduler_type="constant_with_warmup",
            warmup_ratio=0.1,
            optim='adamw_torch',
            weight_decay=0.0,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            save_safetensors=False,
            push_to_hub=False,
            report_to="none"
        )