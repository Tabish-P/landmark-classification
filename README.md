# Landmark Classification

A deep learning project for classifying landmarks using computer vision techniques. This project demonstrates two approaches to landmark classification: building a CNN from scratch and using transfer learning.

## Overview

This project implements landmark classification with 50 different landmark classes. It provides two complementary approaches:

1. **CNN from Scratch**: A custom convolutional neural network built from the ground up
2. **Transfer Learning**: Leveraging pre-trained models (e.g., ResNet-18) for improved performance

## Project Structure

```
landmark-classification/
├── cnn_from_scratch.ipynb      # Notebook for training CNN from scratch
├── transfer_learning.ipynb      # Notebook for transfer learning approach
├── requirements.txt             # Python dependencies
└── src/
    ├── data.py                  # Data loading and preprocessing
    ├── model.py                 # Custom CNN architecture (MyModel)
    ├── transfer.py              # Transfer learning model setup
    ├── train.py                 # Training loop implementation
    ├── optimization.py          # Loss function and optimizer configuration
    ├── predictor.py             # Model inference and prediction
    └── helpers.py               # Utility functions
```

## Features

### Data Processing
- **Data Augmentation**: Random cropping, horizontal flipping, rotation, and grayscale conversion
- **Normalization**: Dataset mean/std normalization
- **Train/Validation/Test Split**: Automatic dataset splitting with configurable validation ratio
- **Batch Processing**: Efficient data loading with PyTorch DataLoaders

### Models

#### Custom CNN (MyModel)
- 4 convolutional blocks with batch normalization and ReLU activation
- Progressive channel expansion: 3 → 64 → 128 → 256 → 512
- MaxPooling for dimensionality reduction
- Adaptive average pooling for fixed-size output
- Fully connected classifier with dropout regularization
- Supports configurable number of classes and dropout rate

#### Transfer Learning
- Pre-trained backbone models (ResNet, VGG, etc.)
- Frozen feature extractor layers
- Custom classification head tailored to landmark task
- Efficient fine-tuning approach

### Training & Optimization
- Configurable optimizers: SGD with momentum or Adam
- Cross-entropy loss for multi-class classification
- GPU support with automatic CUDA detection
- Real-time loss visualization with `livelossplot`
- Progress tracking with tqdm

### Prediction
- PyTorchScript-compatible predictor for deployment
- Automatic input normalization and preprocessing
- Softmax probability outputs
- Inference with gradient disabled

## Requirements

- Python 3.7+
- PyTorch 1.11.0
- TorchVision 0.12.0
- NumPy, Pillow, Matplotlib, OpenCV
- Pandas, Seaborn for analysis
- PyTest for testing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tabish-P/landmark-classification.git
cd landmark-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the environment:
```python
from src.helpers import setup_env
setup_env()
```

## Usage

### Training from Scratch

Open and run `cnn_from_scratch.ipynb`:

```python
from src.data import get_data_loaders
from src.model import MyModel
from src.optimization import get_optimizer, get_loss
from src.train import train_one_epoch

# Load data
data_loaders = get_data_loaders(batch_size=32)

# Initialize model
model = MyModel(num_classes=50, dropout=0.5)

# Set up training
optimizer = get_optimizer(model, optimizer="Adam", learning_rate=0.001)
loss_fn = get_loss()

# Train
for epoch in range(num_epochs):
    train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss_fn)
```

### Transfer Learning

Open and run `transfer_learning.ipynb`:

```python
from src.transfer import get_model_transfer_learning
from src.data import get_data_loaders

# Load pre-trained model
model = get_model_transfer_learning(model_name="resnet18", n_classes=50)

# Load data and train
data_loaders = get_data_loaders(batch_size=32)
```

### Making Predictions

```python
from src.predictor import Predictor
import torch

# Wrap model for inference
predictor = Predictor(model, class_names, mean, std)

# Get predictions
with torch.no_grad():
    predictions = predictor(image_tensor)
```

## Architecture Details

### Custom CNN Architecture
```
Input (3 channels, 224x224)
  ↓
Block 1: Conv2d(3,64) → BN → ReLU → Conv2d(64,64) → BN → ReLU → MaxPool2d (224→112)
  ↓
Block 2: Conv2d(64,128) → BN → ReLU → Conv2d(128,128) → BN → ReLU → MaxPool2d (112→56)
  ↓
Block 3: Conv2d(128,256) → BN → ReLU → Conv2d(256,256) → BN → ReLU → MaxPool2d (56→28)
  ↓
Block 4: Conv2d(256,512) → BN → ReLU → Conv2d(512,512) → BN → ReLU → MaxPool2d (28→14)
  ↓
AdaptiveAvgPool2d (→ 1x1)
  ↓
Classifier: Flatten → Linear(512,512) → BN → ReLU → Dropout(0.5) → Linear(512,num_classes)
```

## Hyperparameters

Common hyperparameters (configurable):

| Parameter | Default | Notes |
|-----------|---------|-------|
| Batch Size | 32 | Mini-batch size for training |
| Learning Rate | 0.001-0.01 | Depends on optimizer and training phase |
| Dropout | 0.5 | Regularization in classifier |
| Validation Split | 0.2 | 20% of data for validation |
| Weight Decay | 0 | L2 regularization (optional) |

## Key Functions

### Data (`src/data.py`)
- `get_data_loaders()`: Create train/validation/test DataLoaders
- `compute_mean_and_std()`: Calculate dataset statistics

### Model (`src/model.py`)
- `MyModel`: Custom CNN class

### Training (`src/train.py`)
- `train_one_epoch()`: Single training epoch
- `validate_one_epoch()`: Validation loop
- `one_epoch_test()`: Test set evaluation

### Transfer Learning (`src/transfer.py`)
- `get_model_transfer_learning()`: Load and configure pre-trained model

### Inference (`src/predictor.py`)
- `Predictor`: Inference wrapper with preprocessing
- `predictor_test()`: Evaluate predictions on test set

## Testing

Run pytest to validate implementations:

```bash
pytest src/
```

Tests cover:
- Model output shapes and types
- Data loader functionality
- Optimizer and loss function setup
- Transfer learning model configuration

## Performance Considerations

- **GPU Acceleration**: Automatically uses CUDA when available
- **Data Augmentation**: Reduces overfitting on small landmark datasets
- **Batch Normalization**: Stabilizes training and allows higher learning rates
- **Adaptive Pooling**: Handles variable input sizes
- **Dropout**: Additional regularization to prevent overfitting

## Next Steps / Enhancements

- Experiment with different architectures (ResNet50, DenseNet, EfficientNet)
- Implement learning rate scheduling and early stopping
- Add cross-validation for more robust evaluation
- Explore ensemble methods combining multiple models
- Implement attention mechanisms for better feature learning

## Dataset

The project expects landmark images organized in train/test/validation folders with class subdirectories. Dataset statistics are automatically computed on first run.

## License

This project is provided as-is for educational purposes.

## Author

Tabish Punjani

---

**Note**: Ensure you have adequate GPU memory when training. The project defaults to GPU if available; CPU training is significantly slower.
