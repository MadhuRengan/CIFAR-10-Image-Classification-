# CIFAR-10 Image Classification using TensorFlow & Keras

An end-to-end deep learning project that builds a Convolutional Neural Network (CNN) from scratch to classify images from the CIFAR-10 dataset into 10 object categories. This project covers the complete machine learning workflow, including data preprocessing, augmentation, model architecture design, training, evaluation, error analysis, and hyperparameter tuning.

✅ Built entirely from scratch (no pre-trained models)  
✅ Efficient TensorFlow/Keras implementation  
✅ Reproducible experiments and systematic tuning  
✅ Improved test accuracy from **85.58% → 87.53%**

---

## Overview

The objective is to classify 32×32 color images into one of the following CIFAR-10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The project focuses on designing a lightweight but effective CNN that balances model complexity, training stability, and generalization performance.

---

## Results

| Model | Batch Size | Dropout | Optimizer | Learning Rate | Test Accuracy |
|---------|---------|---------|---------|---------|---------|
| Baseline | 128 | 0.3 | Adam | 1e-3 | **85.58%** |
| Tuned | 64 | 0.1 | Adam | 1e-3 | **87.53%** |

### Improvement

```text
85.58% → 87.53%
(+1.95% absolute improvement)
```

The performance gain was achieved through targeted hyperparameter tuning without increasing the model size or using transfer learning.

---

## Project Structure

```text
CIFAR-10-Image-Classification/
├── CIFAR-10.ipynb      # complete workflow: preprocessing, training, evaluation, tuning
├── requirements.txt    # Python dependencies
└── README.md
```

Generated during training (not stored in the repository):

```text
baseline_best.keras
tuned_best.keras
```

These files contain the best-performing model checkpoints.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/MadhuRengan/CIFAR-10-Image-Classification-.git

cd CIFAR-10-Image-Classification-
```

### Create and Activate a Virtual Environment

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook CIFAR-10.ipynb
```

The CIFAR-10 dataset will be downloaded automatically the first time the notebook is executed.

---

## Dataset

The project uses the CIFAR-10 dataset provided by TensorFlow:

```python
tensorflow.keras.datasets.cifar10
```

Dataset split:

| Split | Images |
|---------|---------|
| Training | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |

---

## Data Preprocessing

### Normalization

Images are standardized using channel-wise statistics computed only from the training set:

```python
(x - mean) / std
```

This improves optimization stability while preventing data leakage.

### Data Augmentation

Applied exclusively to the training data:

- Random horizontal flip
- Random crop with 4-pixel padding

These augmentations improve generalization by exposing the model to more image variations.

### Efficient Input Pipeline

Implemented using TensorFlow's `tf.data` API:

- Shuffle
- Parallel map
- Batch
- Prefetch

This enables efficient CPU/GPU utilization during training.

---

## Model Architecture

The CNN contains approximately **290K trainable parameters**.

```text
Input (32×32×3)

Stage 1
├── Conv2D (32 filters)
├── BatchNormalization
├── ReLU
├── Conv2D (32 filters)
├── BatchNormalization
├── ReLU
├── MaxPooling
└── Dropout

Stage 2
├── Conv2D (64 filters)
├── BatchNormalization
├── ReLU
├── Conv2D (64 filters)
├── BatchNormalization
├── ReLU
├── MaxPooling
└── Dropout

Stage 3
├── Conv2D (128 filters)
├── BatchNormalization
├── ReLU
├── Conv2D (128 filters)
├── BatchNormalization
├── ReLU
├── MaxPooling
└── Dropout

GlobalAveragePooling
Dense(10, Softmax)
```

### Design Choices

**Convolution + BatchNorm + ReLU**

Each block follows:

```text
Conv → BatchNorm → ReLU
```

Benefits:

- Faster convergence
- More stable training
- Better generalization

Because Batch Normalization already learns a shift parameter, convolution layers use:

```python
use_bias=False
```

to avoid redundant parameters.

**Max Pooling**

Spatial dimensions are reduced as:

```text
32 → 16 → 8 → 4
```

This lowers computation and enables higher-level feature extraction.

**Dropout**

Dropout is applied after each stage to reduce overfitting by randomly deactivating neurons during training.

**Global Average Pooling**

Instead of flattening feature maps, Global Average Pooling summarizes each channel into a single value, significantly reducing parameter count and overfitting risk.

---

## Training Configuration

| Component | Setting |
|------------|----------|
| Loss Function | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Optimizer | Adam |
| Epochs | 30 |
| Initial Learning Rate | 1e-3 |
| Random Seed | 42 |

### Learning Rate Scheduling

```python
ReduceLROnPlateau
```

The learning rate is reduced by a factor of 2 when validation loss does not improve for 3 consecutive epochs.
