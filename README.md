# PSAM: Parallel State-Space and Attention-based Multimodal Framework

## Project Structure
```
.
├── config.py          # Configuration parameters
├── models.py          # Model architecture definitions
├── data_loader.py     # Data loading and preprocessing
├── utils.py           # Utility functions
├── train.py           # Training pipeline
└── main.py            # Main entry point
```


## Usage

1. Configure data path in `config.py`:
```python
DATA_ROOT = "path/to/your/data"
```

2. Run training:
```bash
python main.py
```


## Training Configuration
- Batch size: 32
- Epochs: 200 (with early stopping)
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- LR scheduler: Warmup + Cosine Annealing
- Data split: 80% train / 10% val / 10% test
