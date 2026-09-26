# Plant Disease Classification with CNN (PyTorch)

A convolutional neural network built with PyTorch to classify plant leaf images into 38 disease/health categories using the [PlantVillage dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage).

## Overview

This project trains an image classifier that identifies plant species and disease status from leaf photographs. It's a straightforward CNN trained from scratch (no transfer learning) and reaches **~94.9% test accuracy**.

## Dataset

- **Source:** [PlantVillage dataset on Kaggle](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) (via `kagglehub`)
- **Classes:** 38 plant species/disease categories
- **Train set:** 43,444 images
- **Validation/Test set:** 10,861 images

## Model Architecture

A custom CNN (`MyCNN`) with:

- 3 convolutional blocks (Conv2d → ReLU → BatchNorm → MaxPool), channels growing 3 → 32 → 64 → 128
- A fully connected classifier head (Flatten → Linear(128×16×16 → 128) → ReLU → Dropout(0.4) → Linear(128 → 64) → ReLU → Dropout(0.4) → Linear(64 → num_classes))

Input images are resized to 128×128 and normalized to `[-1, 1]`.

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Batch size | 32 |
| Epochs | 10 |
| Device | CUDA (T4 GPU) |

### Training Loss

| Epoch | Loss |
|---|---|
| 1 | 1.9591 |
| 2 | 1.2570 |
| 3 | 0.9817 |
| 4 | 0.8060 |
| 5 | 0.6606 |
| 6 | 0.5688 |
| 7 | 0.4978 |
| 8 | 0.4502 |
| 9 | 0.4072 |
| 10 | 0.3838 |

## Results

**Test Accuracy: 94.87%**

## Project Structure

```
.
├── plant_disease_classification.ipynb   # Main notebook (data loading, model, training, evaluation)
└── README.md
```

## Requirements

```
torch
torchvision
kagglehub
Pillow
```

## Usage

1. Install dependencies:
   ```bash
   pip install torch torchvision kagglehub pillow
   ```
2. Open and run the notebook in Google Colab or Jupyter (GPU recommended).
3. The dataset is downloaded automatically via `kagglehub.dataset_download("mohitsingh1804/plantvillage")`.
4. Run all cells to train the model and evaluate on the test set.

## Future Improvements

- Add data augmentation (random flips/rotations) to improve generalization
- Try transfer learning (e.g., ResNet, EfficientNet) for higher accuracy
- Add a confusion matrix / per-class metrics for deeper evaluation
- Save and load trained model checkpoints
- Add a simple inference script for single-image predictions
