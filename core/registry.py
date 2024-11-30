import sys
from typing import Dict, Type, Tuple
import inspect

class DNAModelRegistry:
    _implementations: Dict[str, Dict] = {}

    @classmethod
    def register(cls, name: str, head: Type, generator: Type, trainer: Type) -> None:
        cls._implementations[name] = {
            'head': head,
            'generator': generator,
            'trainer': trainer
        }

    @classmethod
    def get_implementation(cls, name: str) -> Tuple[Type, Type, Type]:
        if name not in cls._implementations:
            raise ValueError(f"Implementation {name} not found")
        impl = cls._implementations[name]
        return impl['head'], impl['generator'], impl['trainer']

    @classmethod
    def list_implementations(cls) -> Dict[str, Dict]:
        return cls._implementations

def register_dna_model(implementation_name: str):
    def decorator(head_class):
        module = sys.modules[head_class.__module__]
        generator_class = getattr(module, f"{implementation_name.title()}DataGenerator")
        trainer_class = getattr(module, f"{implementation_name.title()}Trainer")
        
        DNAModelRegistry.register(
            implementation_name,
            head_class,
            generator_class,
            trainer_class
        )
        return head_class
    return decorator