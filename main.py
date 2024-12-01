import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import argparse
import os
import shutil  # Add this import
from pathlib import Path  # Add this import
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoConfig
from dnaCLIP.core.base_classes import BaseDNAModel, BaseTrainer
from dnaCLIP.core.registry import DNAModelRegistry
from datasets import load_dataset

import dnaCLIP.implementations.promoter_prediction
import dnaCLIP.implementations.gc_content
import dnaCLIP.implementations.tm_prediction
import dnaCLIP.implementations.flexibility_prediction
import dnaCLIP.implementations.bedability_prediction

def get_directory_size(path):
    """Calculate total size of a directory in GB"""
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    return total / (1024 * 1024 * 1024)  # Convert to GB

def check_disk_space(path):
    """Check available disk space and current directory size"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Get disk usage information
    total, used, free = shutil.disk_usage(path)
    total_gb = total / (1024 * 1024 * 1024)
    free_gb = free / (1024 * 1024 * 1024)
    used_gb = used / (1024 * 1024 * 1024)
    
    # Get current output directory size if it exists
    current_size = get_directory_size(path) if path.exists() else 0
    
    print("\nDisk Space Information:")
    print(f"Total disk space: {total_gb:.1f} GB")
    print(f"Used disk space: {used_gb:.1f} GB")
    print(f"Free disk space: {free_gb:.1f} GB")
    print(f"Current output directory size: {current_size:.1f} GB")
    
    # Estimate needed space (rough estimate)
    estimated_needed = 2.0  # GB, adjust based on your model size
    print(f"Estimated space needed: {estimated_needed:.1f} GB")
    
    if free_gb < estimated_needed:
        raise RuntimeError(
            f"Not enough disk space! Need {estimated_needed:.1f} GB but only {free_gb:.1f} GB available. "
            "Please free up some space before continuing."
        )
    
    return free_gb > estimated_needed

def list_implementations():
    implementations = DNAModelRegistry.list_implementations()
    print("\nAvailable DNA analysis implementations:")
    print("-" * 50)
    for name, components in implementations.items():
        print(f"\nImplementation: {name}")
        print(f"  Head: {components['head']}")
        print(f"  Generator: {components['generator']}")
        print(f"  Trainer: {components['trainer']}")
    print("\n")

def create_model(implementation: str, model_name: str, dataset_name: str, num_epochs: int = 10):
    # Get implementation components from registry
    head_class, generator_class, trainer_class, test_method = DNAModelRegistry.get_implementation(implementation)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize components with proper config
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        tie_word_embeddings=False,  # Add this to prevent weight sharing
        layer_norm_eps=1e-7,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    backbone = AutoModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,  # Add this to match run2.py
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    head = head_class().to(device)
    data_generator = generator_class()
    
    # Create model and move to device
    model = BaseDNAModel(backbone, head, data_generator).to(device)
    
    # Prepare dataset
    dataset = load_dataset(dataset_name)['train'].train_test_split(test_size=0.1)
    tokenized_dataset = data_generator.prepare_dataset(dataset, tokenizer)
    
    # Create trainer with default arguments
    training_args = BaseTrainer.get_default_args(
        f"outputs/{implementation}",
        num_train_epochs=num_epochs
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_generator.data_collator  # Use the custom data collator
    )
    
    return model, tokenizer, trainer, data_generator, tokenized_dataset, test_method

def main():
    parser = argparse.ArgumentParser(description='DNA Analysis CLI')
    parser.add_argument('-i', '--implementation', type=str, help='Implementation name')
    parser.add_argument('--list', action='store_true', help='List available implementations')
    parser.add_argument('--model', type=str, 
                       default="AIRI-Institute/gena-lm-bert-base-t2t-multi",
                       help='Model name')
    parser.add_argument('--dataset', type=str,
                       default="yurakuratov/example_promoters_300",
                       help='Dataset name')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    if args.list:
        list_implementations()
        return
    
    if not args.implementation:
        print("Please specify an implementation with --implementation or use --list to see available options")
        return
    
    # Check disk space before proceeding
    output_dir = f"outputs/{args.implementation}"
    if not check_disk_space(output_dir):
        return
    
    # Create model, trainer and prepare dataset
    model, tokenizer, trainer, data_generator, tokenized_dataset, test_method = create_model(
        args.implementation, 
        args.model,
        args.dataset,
        args.epochs
    )
    
    # Train model
    trainer.train()
    
    # Run tests
    if test_method:
        print("\nRunning implementation-specific tests...")
        test_method(model, tokenized_dataset["test"], tokenizer)
    elif hasattr(trainer, 'test_model'):
        print("\nRunning trainer's test method...")
        trainer.test_model(tokenized_dataset["test"])
    
    # Save the model
    trainer.save_model(f"outputs/{args.implementation}/final")

if __name__ == "__main__":
    main()