# PyTorch Framework

A comprehensive PyTorch framework for deep learning tasks including classification and segmentation.

## Features

- **Modular Design**: Separate trainers for classification and segmentation tasks
- **Automatic Mixed Precision**: Built-in support for faster training with AMP
- **Comprehensive Metrics**: Accuracy, IoU, Dice coefficient for segmentation
- **Visualization Tools**: Dataset exploration and training progress visualization
- **Configurable**: JSON-based configuration system
- **Checkpointing**: Automatic model saving and resuming

## Project Structure

```
PytorchFramework/
├── configs/                 # Configuration files
│   ├── configs_classification.json
│   └── configs_segmentation.json
├── datasets/               # Dataset classes
│   ├── classification_dataset.py
│   └── segmentation_dataset.py
├── trainers/              # Training classes
│   ├── classification_trainer.py
│   └── segmentation_trainer.py
├── utils/                 # Utility functions
│   ├── helpers.py
│   ├── logger.py
│   ├── metrics.py
│   ├── optimizer_helper.py
│   ├── scheduler_helper.py
│   ├── transforms.py
│   └── visualize.py
├── tests/                 # Unit tests
├── main_train.py          # Main training script
├── predict.py             # Prediction script
├── explore_dataset.py     # Dataset exploration script
└── requirements.txt
```

## Visualization Functions

### Dataset Exploration

Before training, you can explore your dataset to understand the data distribution and verify your data loading pipeline:

```python
# For segmentation datasets
from utils.visualize import explore_segmentation_dataset
explore_segmentation_dataset(
    dataloader=train_loader,
    configs=config,
    num_batches=3,
    samples_per_batch=4,
    class_map=class_map
)

# For classification datasets
from utils.visualize import explore_classification_dataset
explore_classification_dataset(
    dataloader=train_loader,
    configs=config,
    num_batches=3,
    samples_per_batch=4,
    class_map=class_map
)
```

Or use the provided script:
```bash
python explore_dataset.py
```

### Training Visualization

During training, the framework automatically generates:

1. **Metric Curves**: Training and validation metrics over epochs
2. **Prediction Visualizations**: Every 5th epoch shows ALL samples from validation batches
3. **Loss Curves**: Training and validation loss progression

## Usage

### 1. Configuration

Edit the configuration files in `configs/` to match your dataset and training requirements:

```json
{
  "task_type": "segmentation",
  "data": {
    "train_data_dir": "path/to/train/images",
    "train_mask_dir": "path/to/train/masks",
    "val_data_dir": "path/to/val/images", 
    "val_mask_dir": "path/to/val/masks",
    "num_classes": 2,
    "class_map": {"0": "background", "1": "foreground"}
  },
  "training": {
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.001,
    "num_workers": 4
  },
  "output_dir": "outputs/",
  "checkpoints_dir": "checkpoints/"
}
```

### 2. Training

```bash
python main_train.py
```

### 3. Prediction

```bash
python predict.py --model_path checkpoints/best_segmentation_checkpoint.pth --image_path path/to/image.jpg
```

## Key Features

### Segmentation Training
- **Metrics**: Accuracy, IoU (Intersection over Union), Dice coefficient
- **Visualization**: All validation samples visualized every 5th epoch
- **Mixed Precision**: Automatic mixed precision for faster training

### Classification Training  
- **Metrics**: Accuracy
- **Visualization**: Training progress and sample predictions
- **Flexible**: Supports any number of classes

### Visualization Improvements

**Before**: 
- `display_segmentation_batch`: Only showed first batch during training
- `display_segmentation_prediction`: Only showed first batch every 5th epoch

**After**:
- `display_segmentation_batch`: For dataset exploration only (not during training)
- `display_segmentation_prediction`: Shows ALL samples from ALL batches every 5th epoch
- New exploration functions for dataset understanding

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
