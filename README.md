# dnaCLIP

A modular framework for DNA sequence analysis using transformer-based models and CLIP-like architectures.

## Features

- 🧬 Specialized for DNA sequence analysis
- 🔌 Modular plugin architecture
- 🤗 Built on HuggingFace Transformers
- 📊 Multiple prediction tasks support
- 🔄 Easy to extend with new implementations
- 📈 Built-in metrics and visualization
- 🚀 Command-line interface for easy use
- 📝 Automatic logging and evaluation
- 🔍 Detailed testing and metrics reporting

## Supported Tasks

Currently implemented tasks:

- **Promoter Prediction**: Binary classification for promoter identification
  - Metrics: Accuracy, Precision, Recall, F1-score
  - Built-in sequence validation
  - Optimized for batch processing

- **GC Content Prediction**: Regression for GC content estimation
  - Metrics: MSE, MAE, Correlation
  - Sequence-level analysis
  - Real-time evaluation

## Quick Start

### Installation

```bash
git clone https://github.com/ad3002/dnaCLIP.git
cd dnaCLIP
pip install -e .
```

### Command Line Usage

List available implementations:
```bash
python -m dnaCLIP.main --list
```

Train and test a model:
```bash
# For promoter prediction
python -m dnaCLIP.main --implementation promoter -e 5

# For GC content prediction
python -m dnaCLIP.main --implementation gc_content -e 5
```

Custom model and dataset:
```bash
python -m dnaCLIP.main \
    --implementation promoter \
    --model "AIRI-Institute/gena-lm-bert-base-t2t-multi" \
    --dataset "your_dataset_name" \
    -e 10
```

### Python API Usage

```python
from dnaCLIP import DNAModel, DNATrainer, DNADataset

# Promoter prediction
model = DNAModel.from_pretrained("promoter")
predictions = model.predict(["ATCG...", "GCTA..."])

# GC content prediction
gc_model = DNAModel.from_pretrained("gc_content")
gc_values = gc_model.predict(["ATCG...", "GCTA..."])
```

## Configuration

Each implementation supports the following parameters:

```yaml
model:
  name: "AIRI-Institute/gena-lm-bert-base-t2t-multi"
  max_length: 128
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 10
  evaluation_strategy: "steps"
  eval_steps: 100
  logging_steps: 100
```

## Troubleshooting

Common issues and solutions:

1. CUDA out of memory:
   - Reduce batch size
   - Reduce sequence length
   
2. Poor performance:
   - Increase number of epochs
   - Adjust learning rate
   - Check data distribution

3. Data loading issues:
   - Ensure correct dataset format
   - Check sequence length compatibility
   - Verify label format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License

## Citation

```bibtex
@software{dnaclip2024,
  author = {Aleksey Komissarov},
  title = {dnaCLIP: A Modular Framework for DNA Sequence Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ad3002/dnaCLIP}
}
```