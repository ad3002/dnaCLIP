# from run1 import load_model, test_model
# import torch
# torch.cuda.is_available()

# from transformers import AutoTokenizer, AutoModel
# from datasets import load_dataset
# from transformers import DataCollatorWithPadding
# from transformers import TrainingArguments, Trainer
# import numpy as np
# import torch.nn as nn
# import os
# import evaluate
import torch.nn.functional as F

class GCPredictor(nn.Module):
    def __init__(self, gena_lm):
        super().__init__()
        self.sequence_encoder = gena_lm
        
        # Можем начать с замороженной модели
        for param in self.sequence_encoder.parameters():
            param.requires_grad = False
        
        # Простая регрессионная голова
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Так как GC-состав это процент от 0 до 1
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.sequence_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Используем [CLS] токен
        sequence_embedding = outputs.hidden_states[-1][:, 0]
        gc_prediction = self.regressor(sequence_embedding)
        
        return gc_prediction
    
def calculate_gc_content(sequence):
    """Вычисляет реальный GC-состав последовательности"""
    gc_count = sum(1 for base in sequence if base in 'GC')
    return gc_count / len(sequence)

def get_gc_dataset(tokenizer, max_length=128):
    def preprocess_function(examples):
        # Ensure consistent padding and truncation
        tokenized = tokenizer(
            examples["sequence"],
            truncation=True,
            max_length=max_length,
            padding='max_length',  # Changed to force consistent padding
            return_tensors=None    # Ensure we return lists
        )
        
        # Calculate GC content
        tokenized['gc_content'] = [
            calculate_gc_content(seq) for seq in examples["sequence"]
        ]
        
        return tokenized

    dataset = load_dataset("yurakuratov/example_promoters_300")['train'].train_test_split(test_size=0.1)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    return dataset, tokenized_dataset, data_collator

class GCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("gc_content")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        
        loss = F.mse_loss(outputs.squeeze(), labels.float())
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom prediction step to ensure consistent shapes"""
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("gc_content")
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            loss = F.mse_loss(outputs.squeeze(), labels.float())
            
        return (loss.detach(), outputs.detach(), labels)

def train_gc_model(model, tokenized_dataset, data_collator):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
            
        mse = np.mean((predictions - labels) ** 2)
        correlation = np.corrcoef(predictions.flatten(), labels.flatten())[0,1]
        return {
            'mse': mse,
            'correlation': correlation
        }

    training_args = TrainingArguments(
        output_dir="gc_predictor",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        label_names=["gc_content"],
        report_to="none",
    )

    trainer = GCTrainer(  # Changed from Trainer to GCTrainer
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return model

def test_gc_model(dataset, model, tokenizer, num_examples=10):
    """Test GC content prediction model and show sample results"""
    model = model.eval()
    all_predictions = []
    all_labels = []
    all_sequences = []  # Store sequences for later use
    
    # Process test set in batches
    for i in range(0, len(dataset['test']), 32):
        batch = dataset['test'][i:i+32]
        # Only get input_ids and attention_mask
        inputs = tokenizer(
            batch['sequence'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
            model = model.cuda()
        
        with torch.no_grad():
            predictions = model(**model_inputs).numpy().squeeze()
        
        # Get actual GC content
        labels = [calculate_gc_content(seq) for seq in batch['sequence']]
        
        all_predictions.extend(predictions)
        all_labels.extend(labels)
        all_sequences.extend(batch['sequence'])  # Store sequences
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    mse = np.mean((all_predictions - all_labels) ** 2)
    correlation = np.corrcoef(all_predictions, all_labels)[0,1]
    mae = np.mean(np.abs(all_predictions - all_labels))
    
    print("\nOverall Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Correlation: {correlation:.4f}")
    
    print("\nSample Predictions:")
    print("Sequence\t\tPredicted GC\tActual GC\tDiff")
    print("-" * 70)
    
    # Show some sample predictions
    indices = np.random.choice(len(all_predictions), num_examples, replace=False).tolist()  # Convert to Python list
    for idx in indices:
        seq = all_sequences[idx]  # Use stored sequences
        pred = all_predictions[idx]
        actual = all_labels[idx]
        print(f"{seq[:20]}...\t{pred:.3f}\t\t{actual:.3f}\t\t{abs(pred-actual):.3f}")

settings = {
    "model_name": "AIRI-Institute/gena-lm-bert-base-t2t-multi",
    "feature_dim": 128,
    "max_length": 128  # Add this setting
}

tokenizer, gena_lm = load_model(settings)
    
# Create our CLIP-based model
model = GCPredictor(gena_lm)

# Prepare the dataset
dataset, tokenized_dataset, data_collator = get_gc_dataset(tokenizer, settings["max_length"])

# Train the model
model = train_gc_model(model, tokenized_dataset, data_collator)

# Test the model
test_gc_model(dataset, model, tokenizer)