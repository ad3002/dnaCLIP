import torch
import torch.nn as nn
from transformers import DataCollatorWithPadding
from dnaCLIP.core.base_classes import BaseHead, BaseDataGenerator, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from datasets import Dataset, DatasetDict
import pandas as pd

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
    def __init__(self):
        super().__init__()
        self.data_collator = None  # Will be set in prepare_dataset

    def generate_features(self, sequence):
        return sequence
    
    def load_tsv_files(self, train_path, val_path=None, test_path=None):
        """Load dataset from tab-separated files"""
        datasets = {}
        
        # Load train set
        train_df = pd.read_csv(train_path, sep='\t', names=['sequence', 'label'])
        datasets['train'] = Dataset.from_pandas(train_df)
        
        # Load validation set if provided
        if val_path:
            val_df = pd.read_csv(val_path, sep='\t', names=['sequence', 'label'])
            datasets['validation'] = Dataset.from_pandas(val_df)
            
        # Load test set if provided
        if test_path:
            test_df = pd.read_csv(test_path, sep='\t', names=['sequence', 'label'])
            datasets['test'] = Dataset.from_pandas(test_df)
            
        return DatasetDict(datasets)
    
    def prepare_dataset(self, dataset, tokenizer):
        """Prepare dataset for training"""
        if isinstance(dataset, str):
            # If dataset is a file path, load it
            dataset = self.load_tsv_files(dataset)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["sequence"],
                truncation=True,
                max_length=512,  # Add explicit max length
                padding='max_length',  # Changed to fixed-length padding
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
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss for binary classification"""
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
