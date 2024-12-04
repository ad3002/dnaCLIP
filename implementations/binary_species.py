import torch
import torch.nn as nn
from transformers import DataCollatorWithPadding
from dnaCLIP.core.base_classes import BaseHead, BaseDataGenerator, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class BinarySpeciesHead(BaseHead):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
    
    def forward(self, sequence_features, **kwargs):
        # x is the output from BERT's [CLS] token
        return {"logits": torch.sigmoid(self.classifier(sequence_features))}
    
    def compute_loss(self, outputs, targets):
        """Compute binary cross entropy loss"""
        loss_fn = nn.BCELoss()
        return loss_fn(outputs["logits"].squeeze(), targets)
    
    def test(self, sequence_features, **kwargs):
        """Test method for binary classification"""
        outputs = self.forward(sequence_features, **kwargs)
        return (outputs["logits"].squeeze() > 0.5).float()

class BinarySpeciesDataGenerator(BaseDataGenerator):
    def generate_features(self, sequence):
        """Generate features from a DNA sequence"""
        # Not used in this implementation as we use raw sequences
        return sequence
    
    def prepare_dataset(self, dataset, tokenizer):
        """Prepare dataset for training"""
        def tokenize_function(examples):
            return tokenizer(
                examples["sequence"],
                truncation=True,
                padding=False,
                return_tensors=None
            )
        
        def prepare_features(examples):
            examples["labels"] = [float(label) for label in examples["label"]]
            return examples
            
        # Tokenize and prepare features
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.map(prepare_features, batched=True)
        
        # Set format for PyTorch
        tokenized_dataset = tokenized_dataset.remove_columns(['sequence'])
        tokenized_dataset.set_format("torch")
        
        self.data_collator = DataCollatorWithPadding(tokenizer)
        return tokenized_dataset

class BinarySpeciesTrainer(BaseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fn = nn.BCELoss()
        loss = loss_fn(logits.squeeze(), labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def test_model(self, test_dataset):
        """Run model evaluation with detailed metrics"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in self.get_test_dataloader(test_dataset):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = self.model(**batch)
                logits = outputs.get("logits")
                preds = (logits.squeeze() > 0.5).float()
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        print("\nTest Metrics:")
        print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
        print(f"Precision: {precision_score(true_labels, predictions):.4f}")
        print(f"Recall: {recall_score(true_labels, predictions):.4f}")
        print(f"F1 Score: {f1_score(true_labels, predictions):.4f}")

def test_binary_species(model, test_dataset, tokenizer):
    """Additional test function if needed"""
    pass

# Register the implementation
DNAModelRegistry.register(
    "binary_species",
    head=BinarySpeciesHead,
    generator=BinarySpeciesDataGenerator,
    trainer=BinarySpeciesTrainer,
    test_method=test_binary_species
)
