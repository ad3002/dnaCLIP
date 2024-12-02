import trace
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
import traceback


import dnaCLIP.implementations.promoter_prediction
import dnaCLIP.implementations.gc_content
import dnaCLIP.implementations.tm_prediction
import dnaCLIP.implementations.flexibility_prediction
import dnaCLIP.implementations.bendability_prediction
import dnaCLIP.implementations.taxonomy_classification

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

def create_model(implementation: str, model_name: str, dataset_name: str, num_epochs: int = 10, frozen: bool = False, save_checkpoints: bool = True):
    # Get implementation components from registry
    head_class, generator_class, trainer_class, test_method = DNAModelRegistry.get_implementation(implementation)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize components with proper config
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        tie_word_embeddings=False,
        layer_norm_eps=1e-7,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    # Initialize all components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True).to(device)
    head = head_class().to(device)
    data_generator = generator_class()

    # Create the full model
    model = BaseDNAModel(backbone, head, data_generator, frozen=frozen).to(device)
    
    # Prepare dataset with error handling
    try:
        print(f"\nLoading dataset: {dataset_name}")
        raw_dataset = load_dataset(dataset_name)
        if not raw_dataset or 'train' not in raw_dataset:
            raise ValueError(f"Failed to load dataset '{dataset_name}' or missing 'train' split")
        
        print("Splitting dataset...")
        dataset = raw_dataset['train'].train_test_split(test_size=0.1)
        if not dataset or 'train' not in dataset or 'test' not in dataset:
            raise ValueError("Failed to split dataset into train/test")
            
        print("Preparing tokenized dataset...")
        tokenized_dataset = data_generator.prepare_dataset(dataset, tokenizer)
        if not tokenized_dataset:
            raise ValueError("Dataset preparation failed - no data returned from prepare_dataset")
            
        print(f"Dataset preparation complete. Train size: {len(tokenized_dataset['train'])}, "
              f"Test size: {len(tokenized_dataset['test'])}")
    
    except Exception as e:
        print(f"\nError preparing dataset: {str(e)}")
        print("Dataset loading traceback:")
        print(traceback.format_exc())
        raise
    
    # Create trainer with default arguments and properly initialized model
    training_args = BaseTrainer.get_default_args(
        f"outputs/{implementation}",
        num_train_epochs=num_epochs,
        save_checkpoints=save_checkpoints
    )
    
    trainer = trainer_class(
        model=model,  # Now model is defined before being used here
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_generator.data_collator
    )
    
    return model, tokenizer, trainer, data_generator, tokenized_dataset, test_method

def run_pipeline(
    implementation: str,
    model_name: str = "AIRI-Institute/gena-lm-bert-base-t2t-multi",
    dataset_name: str = "yurakuratov/example_promoters_300",
    num_epochs: int = 10,
    frozen: bool = False,
    save_checkpoints: bool = True,
    output_dir: str = None
) -> tuple:
    """
    Run the DNA analysis pipeline programmatically
    
    Returns:
        tuple: (model, tokenizer, trainer, test_results)
    """
    if not output_dir:
        output_dir = f"outputs/{implementation}"
        
    # Check disk space before proceeding
    if not check_disk_space(output_dir):
        raise RuntimeError("Insufficient disk space")
    
    # Create model, trainer and prepare dataset
    model, tokenizer, trainer, data_generator, tokenized_dataset, test_method = create_model(
        implementation, 
        model_name,
        dataset_name,
        num_epochs,
        frozen,
        save_checkpoints
    )
    
    # Train model
    train_results = trainer.train()
    
    # Run tests
    test_results = None
    if test_method:
        print("\nRunning implementation-specific tests...")
        test_results = test_method(model, tokenized_dataset["test"], tokenizer)
    elif hasattr(trainer, 'test_model'):
        print("\nRunning trainer's test method...")
        test_results = trainer.test_model(tokenized_dataset["test"])
    
    # Save the model
    trainer.save_model(f"{output_dir}/final")
    
    return model, tokenizer, trainer, test_results

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
    parser.add_argument('--frozen', action='store_true',
                       help='Freeze BERT backbone weights during training')
    parser.add_argument('--nocheckpoints', action='store_true',
                       help='Disable saving of checkpoints during training')
    args = parser.parse_args()
    
    if args.list:
        list_implementations()
        return
    
    if not args.implementation:
        print("Please specify an implementation with --implementation or use --list to see available options")
        return
    
    try:
        model, tokenizer, trainer, test_results = run_pipeline(
            implementation=args.implementation,
            model_name=args.model,
            dataset_name=args.dataset,
            num_epochs=args.epochs,
            frozen=args.frozen,
            save_checkpoints=not args.nocheckpoints
        )
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error running pipeline: {str(e)}")
        return

if __name__ == "__main__":
    main()