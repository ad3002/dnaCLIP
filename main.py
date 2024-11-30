import argparse
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from .core.base_classes import BaseDNAModel, BaseTrainer
from .core.registry import DNAModelRegistry
from datasets import load_dataset

# Import all implementations to register them
from .implementations import promoter_prediction, gc_content, splice_sites

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

def create_model(implementation: str, model_name: str, dataset_name: str):
    # Get implementation components from registry
    head_class, generator_class, trainer_class = DNAModelRegistry.get_implementation(implementation)
    
    # Initialize components
    backbone = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    head = head_class()
    data_generator = generator_class()
    
    # Create model
    model = BaseDNAModel(backbone, head, data_generator)
    
    # Prepare dataset
    dataset = load_dataset(dataset_name)['train'].train_test_split(test_size=0.1)
    tokenized_dataset = data_generator.prepare_dataset(dataset, tokenizer)
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer with default arguments
    training_args = BaseTrainer.get_default_args(f"outputs/{implementation}")
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    return model, tokenizer, trainer, data_generator, tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description='DNA Analysis CLI')
    parser.add_argument('--implementation', type=str, help='Implementation name')
    parser.add_argument('--list', action='store_true', help='List available implementations')
    parser.add_argument('--model', type=str, 
                       default="AIRI-Institute/gena-lm-bert-base-t2t-multi",
                       help='Model name')
    parser.add_argument('--dataset', type=str,
                       default="yurakuratov/example_promoters_300",
                       help='Dataset name')
    args = parser.parse_args()
    
    if args.list:
        list_implementations()
        return
    
    if not args.implementation:
        print("Please specify an implementation with --implementation or use --list to see available options")
        return
    
    # Create model, trainer and prepare dataset
    model, tokenizer, trainer, data_generator, tokenized_dataset = create_model(
        args.implementation, 
        args.model,
        args.dataset
    )
    
    # Train model
    trainer.train()
    
    # Save the model
    trainer.save_model(f"outputs/{args.implementation}/final")

if __name__ == "__main__":
    main()