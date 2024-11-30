# dnaCLIP

A modular framework for DNA sequence analysis using transformer-based models and CLIP-like architectures.

## Features

- 🧬 Specialized for DNA sequence analysis
- 🔌 Modular plugin architecture
- 🤗 Built on HuggingFace Transformers
- 📊 Multiple prediction tasks support
- 🔄 Easy to extend with new implementations
- 📈 Built-in metrics and visualization

## Supported Tasks

Currently implemented tasks:

- **Promoter Prediction**: Binary classification of DNA sequences for promoter presence
- **GC Content Prediction**: Regression task for GC content estimation

## Quick Start

### Installation

```bash
git clone https://github.com/ad3002/dnaCLIP.git
cd dnaCLIP
pip install -e .
```

### Basic Usage

```python
from dnaCLIP import DNAModel

# Initialize model for promoter prediction
model = DNAModel.from_pretrained("promoter")

# Make predictions
sequences = ["ATCG...", "GCTA..."]
predictions = model.predict(sequences)
```

## Advanced Usage

### Training a New Model

```python
from dnaCLIP import DNATrainer, DNADataset

# Prepare dataset
dataset = DNADataset.from_files(
    train="train.fa",
    test="test.fa",
    task="promoter"
)

# Initialize and train
trainer = DNATrainer(
    implementation="promoter",
    model="AIRI-Institute/gena-lm-bert-base-t2t-multi"
)
trainer.train(dataset)
```

### Custom Implementation

1. Create implementation file:

```python
from dnaCLIP.core import BaseHead, BaseTrainer, BaseDataGenerator

class MyTaskHead(BaseHead):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(...)
    
    def forward(self, x):
        return self.net(x)

# Register implementation
DNAModelRegistry.register(
    "my_task",
    MyTaskHead,
    MyTaskDataGenerator,
    MyTaskTrainer,
    test_function
)
```

2. Use your implementation:

```python
model = DNAModel.from_pretrained("my_task")
```

## Configuration

Each implementation can be configured via a config file:

```yaml
model:
  name: "AIRI-Institute/gena-lm-bert-base-t2t-multi"
  max_length: 512

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 10
```

## Testing

Run test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License

## Citation

If you use dnaCLIP in your research, please cite:

```bibtex
@software{dnaclip2024,
  author = {Aleksey Komissarov},
  title = {dnaCLIP: A Modular Framework for DNA Sequence Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ad3002/dnaCLIP}
}
```