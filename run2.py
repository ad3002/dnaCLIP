import torch
torch.cuda.is_available()

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np
import torch.nn as nn
import os
import evaluate

settings = {
    "model_name": "AIRI-Institute/gena-lm-bert-base-t2t-multi",
    "feature_dim": 128,
    "max_length": 128  # Add this setting
}


import torch.nn.functional as F


class SpliceSiteCLIP(nn.Module):
    def __init__(self, gena_lm, feature_dim=128):
        super().__init__()
        self.sequence_encoder = gena_lm
        
        # Unfreeze GENA-LM weights to allow fine-tuning
        for param in self.sequence_encoder.parameters():
            param.requires_grad = True  # Change to True
        
        self.sequence_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, feature_dim)
        )
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(5, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, feature_dim)
        )
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    
    def forward(self, input_ids, attention_mask, features):
        sequence_outputs = self.sequence_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the hidden states from the output
        # Use the last layer's [CLS] token representation
        sequence_embedding = sequence_outputs.hidden_states[-1][:, 0]
        sequence_embedding = self.sequence_projector(sequence_embedding)
        
        feature_embedding = self.feature_encoder(features)
        
        sequence_embedding = F.normalize(sequence_embedding, dim=-1)
        feature_embedding = F.normalize(feature_embedding, dim=-1)
        
        return sequence_embedding, feature_embedding


metric = evaluate.load("accuracy")
    
def compute_metrics(eval_pred):
    logits_and_embeddings, labels = eval_pred
    if logits_and_embeddings is None:
        return {"eval_accuracy": 0.0, "eval_loss": float('inf')}
    
    sequence_embeddings, feature_embeddings = logits_and_embeddings
    
    # Normalize embeddings
    sequence_embeddings = F.normalize(sequence_embeddings, dim=-1)
    feature_embeddings = F.normalize(feature_embeddings, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(sequence_embeddings, feature_embeddings.T)
    similarity = similarity / 0.07  # Use the same temperature
    
    # Predictions and labels
    predictions = similarity.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Compute accuracy
    accuracy = (predictions == labels).mean()
    loss = F.cross_entropy(similarity, torch.from_numpy(labels).to(similarity.device)).item()
    
    return {
        "eval_accuracy": accuracy,
        "eval_loss": loss
    }

def compute_clip_loss(model, sequence_embeddings, feature_embeddings):
    # Compute similarity matrix
    similarity = torch.matmul(sequence_embeddings, feature_embeddings.T)
    
    # Apply temperature scaling
    similarity = similarity / model.temperature
    
    # Use labels from 0 to batch_size - 1
    batch_size = sequence_embeddings.size(0)
    labels = torch.arange(batch_size).to(sequence_embeddings.device)
    
    # Compute loss in both directions
    loss_sequence = F.cross_entropy(similarity, labels)
    loss_feature = F.cross_entropy(similarity.T, labels)
    
    return (loss_sequence + loss_feature) / 2

class CLIPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Only save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sequence_embeddings, feature_embeddings = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            features=inputs['features']
        )
        
        loss = compute_clip_loss(model, sequence_embeddings, feature_embeddings)
        
        if return_outputs:
            return loss, (sequence_embeddings, feature_embeddings)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction_step to return embeddings and compute loss for evaluation.
        """
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")  # Get labels from inputs
        
        with torch.no_grad():
            sequence_embeddings, feature_embeddings = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                features=inputs['features']
            )
            
            # Compute the loss
            loss = compute_clip_loss(model, sequence_embeddings, feature_embeddings)
            
            # Move embeddings to CPU
            sequence_embeddings = sequence_embeddings.cpu()
            feature_embeddings = feature_embeddings.cpu()
            if labels is not None:
                labels = labels.cpu()
            
        return loss.cpu(), (sequence_embeddings, feature_embeddings), labels


class CLIPDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        # Use the sample indices as labels to ensure consistency across epochs
        batch_size = len(features)
        batch["labels"] = torch.arange(batch_size)
        return batch

def train_model(model, tokenizer, tokenized_dataset, data_collator):
    training_args = TrainingArguments(
        output_dir="splice_clip_model",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_safetensors=False,
        push_to_hub=False,
        report_to="none",
        metric_for_best_model="eval_accuracy",  # Add this line
        greater_is_better=True,  # Add this line
        label_names=["labels"],  # Add this line
    )

    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model


settings = {
    "model_name": "AIRI-Institute/gena-lm-bert-base-t2t-multi",
    "feature_dim": 128,  # Dimension of our shared embedding space
    "max_length": 128  # Add this setting
}

def load_model(settings):
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"])
    # Initialize model with tie_weights=False to prevent weight sharing
    gena_lm = AutoModel.from_pretrained(
        settings["model_name"],
        trust_remote_code=True,
        tie_word_embeddings=False  # Add this parameter
    )
    return tokenizer, gena_lm

def create_splice_clip_model(gena_lm, feature_dim):
    # New function to create our CLIP model
    model = SpliceSiteCLIP(
        gena_lm=gena_lm,
        feature_dim=feature_dim
    )
    return model

def extract_genomic_features(sequence):
    # Temporary simple features for demonstration
    gc_content = sum(1 for base in sequence if base in 'GC') / len(sequence)
    tata_boxes = sum(1 for i in range(len(sequence) - 5) if sequence[i:i+5] == 'TATAA')
    # Add more feature calculations as needed
    features = torch.tensor([
        gc_content,
        tata_boxes,
        0.0,  # placeholder for motif strength
        0.0,  # placeholder for secondary structure
        0.0,  # placeholder for position bias
    ], dtype=torch.float32)
    return features

def get_dataset(tokenizer):
    # Modified to handle splice site data
    
    
    def preprocess_function(examples):
        # Tokenize sequences with fixed padding
        tokenized = tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=settings["max_length"],
            padding="max_length",  # Change to max_length
            return_tensors=None    # Add this
        )
        
        # Add genomic features
        features = [extract_genomic_features(seq) for seq in examples["sequence"]]
        tokenized['features'] = torch.stack(features)
        
        return tokenized

    # For now, we'll use the same dataset but treat it as splice sites
    # In practice, you'd want to create a proper splice site dataset
    dataset = load_dataset("yurakuratov/example_promoters_300")['train'].train_test_split(test_size=0.1)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = CLIPDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=settings["max_length"]
    )
    
    return dataset, tokenized_dataset, data_collator

def test_model(dataset, model, tokenizer):
    model = model.eval()
    all_accuracies = []
    
    for i in range(0, len(dataset['test']), 32):  # Process in batches
        batch = dataset['test'][i:i+32]
        
        # Prepare inputs
        inputs = tokenizer(batch['sequence'], return_tensors='pt', padding=True)
        features = torch.stack([
            extract_genomic_features(seq) for seq in batch['sequence']
        ])
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            features = features.cuda()
            model = model.cuda()
        
        with torch.no_grad():
            sequence_embeddings, feature_embeddings = model(**inputs, features=features)
            
            # Compute similarity matrix
            similarity = torch.matmul(sequence_embeddings, feature_embeddings.T)
            predictions = torch.argmax(similarity, dim=1)
            
            # Compute accuracy (diagonal elements should match)
            labels = torch.arange(len(predictions)).to(predictions.device)
            accuracy = (predictions == labels).float().mean()
            all_accuracies.append(accuracy.item())
    
    final_accuracy = np.mean(all_accuracies)
    print(f'Average matching accuracy: {final_accuracy:.4f}')

# if __name__ == '__main__':
#     # Load the base GENA-LM model and tokenizer
#     tokenizer, gena_lm = load_model(settings)
    
#     # Create our CLIP-based model
#     model = create_splice_clip_model(gena_lm, settings['feature_dim'])
    
#     # Prepare the dataset
#     dataset, tokenized_dataset, data_collator = get_dataset(tokenizer)
    
#     # Train the model
#     model = train_model(model, tokenized_dataset, data_collator)
    
#     # Test the model
#     test_model(dataset, model, tokenizer)