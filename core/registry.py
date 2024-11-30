from typing import Dict, Type, Tuple, Optional, Callable

class DNAModelRegistry:
    _implementations: Dict[str, Dict] = {}

    @classmethod
    def register(cls, 
                 name: str, 
                 head: Type, 
                 generator: Type, 
                 trainer: Type,
                 test_method: Optional[Callable] = None) -> None:
        cls._implementations[name] = {
            'head': head,
            'generator': generator,
            'trainer': trainer,
            'test_method': test_method
        }

    @classmethod
    def get_implementation(cls, name: str) -> Tuple[Type, Type, Type, Optional[Callable]]:
        if name not in cls._implementations:
            raise ValueError(f"Implementation {name} not found")
        impl = cls._implementations[name]
        return impl['head'], impl['generator'], impl['trainer'], impl['test_method']

    @classmethod
    def list_implementations(cls) -> Dict[str, Dict]:
        return cls._implementations