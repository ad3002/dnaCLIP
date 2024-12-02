from abc import ABC, abstractmethod
import torch
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

class BaseDNAModel(nn.Module):
    def __init__(self, backbone, head, data_generator, frozen=False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.data_generator = data_generator
        self.frozen = frozen
        
        # Set backbone to frozen state if requested
        if self.frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        # Use no_grad for backbone if frozen
        if self.frozen:
            with torch.no_grad():
                backbone_output = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        else:
            backbone_output = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Handle different types of model outputs
        if hasattr(backbone_output, 'hidden_states'):
            # Get last hidden state from hidden_states tuple
            sequence_output = backbone_output.hidden_states[-1]
        elif hasattr(backbone_output, 'last_hidden_state'):
            sequence_output = backbone_output.last_hidden_state
        elif isinstance(backbone_output, dict):
            sequence_output = backbone_output.get('last_hidden_state', 
                                               backbone_output.get('hidden_states', None))
        elif isinstance(backbone_output, tuple):
            sequence_output = backbone_output[0]
        elif isinstance(backbone_output, torch.Tensor):
            sequence_output = backbone_output
        else:
            raise ValueError(f"Unexpected backbone output type: {type(backbone_output)}\nOutput: {backbone_output}")
        
        if sequence_output is None:
            raise ValueError(f"Could not extract sequence output from backbone model. Output: {backbone_output}")
        
        # Extract CLS token representation
        sequence_features = sequence_output[:, 0, :]
        
        # Pass through head
        return self.head(sequence_features)
    
    def unfreeze(self):
        """Unfreeze the backbone parameters"""
        self.frozen = False
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze(self):
        """Freeze the backbone parameters"""
        self.frozen = True
        for param in self.backbone.parameters():
            param.requires_grad = False

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
    def get_default_args(output_dir: str, num_train_epochs: int = 10, nocheckpoint: bool = False) -> TrainingArguments:
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
            save_strategy="no" if nocheckpoint else "epoch",
            logging_strategy="epoch",
            load_best_model_at_end=not nocheckpoint,
            save_safetensors=False,
            push_to_hub=False,
            report_to="none"
        )