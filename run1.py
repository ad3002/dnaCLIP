import torch
torch.cuda.is_available()

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import numpy as np
from torch import nn

settings = {
    "model_name": "'AIRI-Institute/gena-lm-bert-base-t2t-multi",
}

def load_model(settings):
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"])
    model = AutoModel.from_pretrained(settings["model_name"], trust_remote_code=True)
    gena_module_name = model.__class__.__module__
    return tokenizer, model, gena_module_name

def demo(tokenizer, model):

    seq = 'CACCCAGAGAGAGTAACCAGAATGGATACATTTTGGCCAACATGATTCTAACCCAGTGAGACCCATTTTGGGCTTATG'
    tokens = tokenizer.tokenize(seq, add_special_tokens=True)
    print('tokens:', tokens)
    print('n_tokens:', len(tokens))
    with torch.no_grad():
        output = model(**tokenizer(seq, return_tensors='pt'), output_hidden_states=True)
    print(output.keys())
    print(output['hidden_states'][-1].shape)

def get_extended_model(settings, model, n_classes=2):
    import importlib
    # available class names:
    # - BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
    # - BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
    # - BertForQuestionAnswering
    # check https://huggingface.co/docs/transformers/model_doc/bert
    tokenizer, model, gena_module_name = load_model(settings)
    print(gena_module_name)
    cls = getattr(importlib.import_module(gena_module_name), 'BertForSequenceClassification')
    print(cls)

    model = cls.from_pretrained(settings["model_name"], num_labels=n_classes)
    print('\nclassification head:', model.classifier)
    return model

def get_dataset(tokenizer):

    def preprocess_labels(example):
        example['label'] = example['promoter_presence']
        return example
    
    def preprocess_function(examples):
        # just truncate right, but for some tasks symmetric truncation from left and right is more reasonable
        # set max_length to 128 tokens to make experiments faster
        return tokenizer(examples["sequence"], truncation=True, max_length=128)
    
    # load ~11k samples from promoters prediction dataset
    dataset = load_dataset("yurakuratov/example_promoters_300")['train'].train_test_split(test_size=0.1)
    # print(dataset)
    # print(dataset['train'][0])
    # print('# base pairs: ', len(dataset['train'][0]['sequence']))
    # print('tokens: ', ' '.join(tokenizer.tokenize(dataset['train'][0]['sequence'])))
    # print('# tokens: ', len(tokenizer.tokenize(dataset['train'][0]['sequence'])))
    dataset = dataset.map(preprocess_labels)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return dataset, tokenized_dataset, data_collator


def train_model(model, tokenized_dataset, data_collator):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': (predictions==labels).sum() / len(labels)}

    # change training hyperparameters to archive better quality
    training_args = TrainingArguments(
        output_dir="test_run",
        learning_rate=2e-05,
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
        load_best_model_at_end=True
    )

    trainer = Trainer(
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


def test_model(dataset, model, tokenizer):
    model = model.eval()
    all_predictions = []
    all_labels = []

    for i in range(len(dataset['test'])):
        x, y = dataset['test']['sequence'][i], dataset['test']['label'][i]
        x_feat = tokenizer(x, return_tensors='pt')
        
        # move sample to gpu and feed it to the model
        for k in x_feat:
            x_feat[k] = x_feat[k].cuda()

        with torch.no_grad():
            out = model(**x_feat)

        # get class probabilities
        prob = torch.softmax(out['logits'], dim=-1)
        prediction = torch.argmax(prob).item()

        all_predictions.append(prediction)
        all_labels.append(y)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = (all_predictions == all_labels).sum() / len(all_labels)
    print(f'Accuracy: {accuracy}')

def predict_one(x, model, tokenizer):
    model = model.eval()
    x_feat = tokenizer(x, return_tensors='pt')
    
    # move sample to gpu
    for k in x_feat:
        x_feat[k] = x_feat[k].cuda()

    with torch.no_grad():
        out = model(**x_feat)

    prob = torch.softmax(out['logits'], dim=-1)
    prediction = torch.argmax(prob).item()
    return prediction

import matplotlib.pyplot as plt

def bin_enrichment(s, model, tokenizer, bin_size=50):
    total_bins = len(s) // bin_size + 1
    middle_bin = total_bins // 2
    bin_counts = [0] * total_bins
    
    for i in range(len(s) - 300 + 1):
        x = s[i : i + 300]
        prediction = predict_one(x, model, tokenizer)
        bin_index = i // bin_size
        bin_counts[bin_index] += prediction

    # Normalize bin counts
    bin_windows = [0] * total_bins
    for i in range(len(s) - 300 + 1):
        bin_index = i // bin_size
        bin_windows[bin_index] += 1

    bin_enrichment = [count / windows if windows > 0 else 0 for count, windows in zip(bin_counts, bin_windows)]
    
    # Create centered x-axis labels
    x_labels = range(-middle_bin, total_bins - middle_bin)
    
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.bar(x_labels, bin_enrichment)
    plt.xlabel("Bin Number (0 = center)")
    plt.ylabel("Promoter Enrichment")
    plt.title(f"Promoter Enrichment Across {bin_size} bp Bins")
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)  # Add vertical line at center
    plt.show()


if __name__ == '__main__':
    tokenizer, model, gena_module_name = load_model(settings)
    demo(tokenizer, model)
    model = get_extended_model(settings, model)
    dataset, tokenized_dataset, data_collator = get_dataset(tokenizer)
    model = train_model(model, tokenized_dataset, data_collator)
    test_model(dataset, model, tokenizer)