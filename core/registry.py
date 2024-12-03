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

    @classmethod
    def get_implementation_classes(cls, implementation):
        """Get head and generator classes for an implementation
        
        Args:
            implementation (str): The name of the implementation
            
        Returns:
            tuple: (head_class, generator_class) or (None, None) if not found
        """
        if implementation not in cls._implementations:
            return None, None
        return cls._implementations[implementation]['head'], cls._implementations[implementation]['generator']
    
    @classmethod
    def get_head_class(cls, head_type):
        """Get head class by its type name"""
        for impl in cls._implementations.values():
            if impl['head'].__name__ == head_type:
                return impl['head']
        return None

    @classmethod
    def get_generator_class(cls, head_type):
        """Get generator class associated with a head type"""
        for impl in cls._implementations.values():
            if impl['head'].__name__ == head_type:
                return impl['generator']
        return None