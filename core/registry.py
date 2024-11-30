
from typing import Dict, Type, Tuple
from .base_classes import BaseDataGenerator, BaseHead, BaseTrainer

class DNAModelRegistry:
    _heads: Dict[str, Type[BaseHead]] = {}
    _generators: Dict[str, Type[BaseDataGenerator]] = {}
    _trainers: Dict[str, Type[BaseTrainer]] = {}
    
    @classmethod
    def register_implementation(cls, name: str, head: Type[BaseHead], 
                             generator: Type[BaseDataGenerator],
                             trainer: Type[BaseTrainer]):
        """Register a complete implementation (head, generator, and trainer)."""
        cls._heads[name] = head
        cls._generators[name] = generator
        cls._trainers[name] = trainer
    
    @classmethod
    def get_implementation(cls, name: str) -> Tuple[Type[BaseHead], 
                                                   Type[BaseDataGenerator],
                                                   Type[BaseTrainer]]:
        """Get implementation components by name."""
        if name not in cls._heads:
            raise ValueError(f"Implementation '{name}' not found")
        return cls._heads[name], cls._generators[name], cls._trainers[name]
    
    @classmethod
    def list_implementations(cls) -> Dict[str, Dict]:
        """List all registered implementations with their components."""
        return {
            name: {
                "head": cls._heads[name].__name__,
                "generator": cls._generators[name].__name__,
                "trainer": cls._trainers[name].__name__
            }
            for name in cls._heads.keys()
        }

def register_dna_model(name: str):
    """Decorator to register a set of implementations."""
    def decorator(head_class: Type[BaseHead]):
        # Assume generator and trainer classes are in the same module with similar names
        module = head_class.__module__
        module_parts = module.split('.')
        implementation_name = module_parts[-1]
        
        # Import related classes
        import importlib
        module = importlib.import_module(module)
        generator_class = getattr(module, f"{implementation_name.title()}DataGenerator")
        trainer_class = getattr(module, f"{implementation_name.title()}Trainer")
        
        # Register implementation
        DNAModelRegistry.register_implementation(
            name,
            head_class,
            generator_class,
            trainer_class
        )
        return head_class
    return decorator