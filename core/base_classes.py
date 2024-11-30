from abc import ABC, abstractmethod
import torch.nn as nn
from transformers import Trainer, TrainingArguments, PreTrainedModel

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
    
    @abstractmethod
    def test(self, sequence_features, **kwargs):
        """Test method for head-specific testing logic"""
        pass

class BaseDNAModel(PreTrainedModel):
    def __init__(self, backbone, head, data_generator):
        super().__init__(backbone.config)
        self.backbone = backbone
        self.head = head
        self.data_generator = data_generator
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.backbone(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,  # Request hidden states explicitly
            return_dict=True
        )
        # Get the last hidden state - works for both output types
        if hasattr(outputs, 'last_hidden_state'):
            sequence_features = outputs.last_hidden_state[:, 0, :]
        else:
            # For models that return hidden_states differently
            sequence_features = outputs.hidden_states[-1][:, 0, :]
        return self.head(sequence_features, **kwargs)

class BaseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Handle data collator
        if 'data_collator' not in kwargs and hasattr(kwargs.get('model', None), 'data_generator'):
            kwargs['data_collator'] = kwargs['model'].data_generator.data_collator
        
        # Handle tokenizer/processing_class
        if 'tokenizer' in kwargs:
            if 'processing_class' not in kwargs:
                kwargs['processing_class'] = kwargs['tokenizer']
            kwargs.pop('tokenizer')  # Always remove tokenizer to avoid conflicts
            
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation"""
        pass
    
    @abstractmethod
    def test_model(self, test_dataset, num_examples=10):
        """Run detailed testing on the model"""
        pass
    
    @abstractmethod
    def get_test_metrics(self, predictions, labels):
        """Calculate test-specific metrics"""
        pass
    
    @staticmethod
    def get_default_args(output_dir: str, num_train_epochs: int = 10) -> TrainingArguments:
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
            num_train_epochs=num_train_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            save_safetensors=False,
            push_to_hub=False,
            report_to="none"
        )