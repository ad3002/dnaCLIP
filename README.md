# dnaCLIP

A modular framework for DNA sequence analysis using CLIP-like architectures and transformer-based models.

## Overview

dnaCLIP is a flexible framework that allows implementation of various DNA sequence analysis tasks using a common architecture based on transformer models. It currently supports:

- Promoter prediction
- GC content prediction

## Installation

```bash
git clone https://github.com/yourusername/dnaCLIP.git
cd dnaCLIP
pip install -r requirements.txt
```

## Project Structure

```
dnaCLIP/
├── core/
│   ├── base_classes.py    # Abstract base classes
│   └── registry.py        # Implementation registry system
├── implementations/
│   ├── promoter_prediction.py
│   └── gc_content.py
└── main.py               # CLI interface
```

## Usage

List available implementations:
```bash
python main.py --list
```

Train a model:
```bash
python main.py --implementation promoter
python main.py --implementation gc_content
```

Custom dataset and model:
```bash
python main.py --implementation promoter \
               --model "your/model" \
               --dataset "your/dataset"
```

## Adding New Implementations

1. Create a new file in `implementations/`
2. Implement three classes:
   - `YourTaskDataGenerator(BaseDataGenerator)`
   - `YourTaskHead(BaseHead)`
   - `YourTaskTrainer(BaseTrainer)`
3. Add `@register_dna_model("your_task")` decorator to the Head class

Example:
```python
@register_dna_model("new_task")
class NewTaskHead(BaseHead):
    def __init__(self, input_dim=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, n_classes)
        )
    # ... implement required methods
```

## Base Models

The framework uses GENA-LM as the default backbone but supports any HuggingFace transformer model:
- Default: "AIRI-Institute/gena-lm-bert-base-t2t-multi"
- Compatible with any BERT-like architecture

## Features

- Modular architecture with easy extension
- Automatic implementation registration
- Built on HuggingFace Transformers
- Unified training interface
- Customizable data processing
- Multiple task support

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dnaclip2024,
  author = {Aleksey Komissarov},
  title = {dnaCLIP: A Framework for DNA Sequence Analysis},
  year = {2024},
  url = {https://github.com/ad3002/dnaCLIP}
}
```