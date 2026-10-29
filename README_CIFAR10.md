# CIFAR-10 Image Classification with TensorFlow and Keras

A convolutional neural network (CNN) project for classifying 32 × 32 RGB images into the 10 CIFAR-10 categories. The project documentation describes a CNN trained from scratch, with data augmentation, a TensorFlow input pipeline, checkpointing, and hyperparameter tuning.

> **Reproducibility status:** As inspected on 29 October 2026, the committed `CIFAR-10.ipynb` file is empty (0 bytes). The architecture, training settings, and metrics below are taken from the existing project README; they cannot be verified against runnable training code until the notebook is restored.

## Project overview

The task is to predict one of 10 labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck. The approach uses a lightweight CNN built from scratch rather than a pretrained backbone.

**Documented results**

| Experiment | Batch size | Dropout | Optimizer | Initial learning rate | Reported test accuracy |
| --- | ---: | ---: | --- | ---: | ---: |
| Baseline | 128 | 0.3 | Adam | 0.001 | **85.58%** |
| Tuned | 64 | 0.1 | Adam | 0.001 | **87.53%** |

The reported gain is **1.95 percentage points** in test accuracy. These figures are from the existing README, not newly reproduced measurements.

## Project structure

```text
CIFAR-10-Image-Classification-/
├── CIFAR-10.ipynb       # Intended end-to-end experiment notebook; currently empty in the repository
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
```

The existing documentation also describes two locally generated checkpoints, which are **not tracked in the repository**:

```text
baseline_best.keras
tuned_best.keras
```

These represent the best model checkpoints for the baseline and tuned runs, respectively, according to the documented workflow.

## Dataset and preprocessing

The project uses the CIFAR-10 dataset distributed via `tensorflow.keras.datasets.cifar10`, containing 60,000 color images of size 32 × 32 pixels across 10 classes.

| Split | Images |
| --- | ---: |
| Training | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |

The existing README describes the following preprocessing setup:

- **Channel-wise standardization:** normalize inputs as `(image - training_mean) / training_std`, calculating statistics from the training partition to avoid leakage.
- **Training-only augmentation:** random horizontal flips and random crops with four-pixel padding to increase visual variety.
- **`tf.data` pipeline:** shuffle, parallel map, batch, and prefetch to reduce input bottlenecks.

The validation and test sets should use deterministic preprocessing without random augmentation.

## Model architecture

The documented model is a custom CNN with approximately **290K trainable parameters** and three stages of increasing channel depth.

```text
Input: 32 × 32 × 3 RGB image
    │
    ├── Stage 1: Conv(32) → BatchNorm → ReLU
    │            Conv(32) → BatchNorm → ReLU
    │            MaxPool → Dropout
    │
    ├── Stage 2: Conv(64) → BatchNorm → ReLU
    │            Conv(64) → BatchNorm → ReLU
    │            MaxPool → Dropout
    │
    ├── Stage 3: Conv(128) → BatchNorm → ReLU
    │            Conv(128) → BatchNorm → ReLU
    │            MaxPool → Dropout
    │
    └── GlobalAveragePooling → Dense(10, softmax)
```

### Model design decisions

| Decision | Rationale |
| --- | --- |
| **Convolutional layers** | Learn local visual patterns such as edges, textures, and object parts. |
| **Increasing filters (32 → 64 → 128)** | Build richer representations as spatial resolution decreases. |
| **Batch normalization followed by ReLU** | Stabilize intermediate activations and introduce nonlinearity. |
| **`use_bias=False` in convolutions** | Avoid a redundant convolution bias when batch normalization provides a learned offset. |
| **Max pooling after each stage** | Reduce spatial dimensions from 32 → 16 → 8 → 4, lowering computation. |
| **Dropout after each stage** | Regularize the network to mitigate overfitting. |
| **Global average pooling** | Replace a parameter-heavy flatten/dense head with one aggregated value per feature channel. |
| **10-unit softmax output** | Return class probabilities for the 10 mutually exclusive categories. |

The model does not rely on transfer learning or pretrained weights.

## Training setup

The existing project documentation specifies the following configuration:

| Component | Setting |
| --- | --- |
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Evaluation metric | Accuracy |
| Initial learning rate | `1e-3` |
| Epochs | 30 |
| Random seed | 42 |
| Learning-rate schedule | `ReduceLROnPlateau` |
| Plateau monitoring | Validation loss |
| Scheduler patience | 3 epochs |
| Learning-rate reduction factor | 0.5 |

### Hyperparameter experiments

The documented baseline uses **batch size 128** and **dropout 0.3**. The tuned run uses **batch size 64** and **dropout 0.1**, keeping Adam and the initial learning rate unchanged. The existing README reports test accuracy increasing from **85.58% to 87.53%**.

These are reported experimental settings and results. Checkpoint paths, exact callback ordering, train/validation split logic, and evaluation scripts need confirmation once the notebook source is restored.

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/MadhuRengan/CIFAR-10-Image-Classification-.git
cd CIFAR-10-Image-Classification-
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
tensorflow>=2.13
numpy
matplotlib
jupyter
```

A compatible Python version and TensorFlow installation are required. A GPU is optional but can accelerate training.

### 4. Restore the notebook, then launch Jupyter

**Important:** The `CIFAR-10.ipynb` file committed on `main` is currently empty. Restore or upload the actual training notebook before attempting to execute it.

```bash
jupyter notebook CIFAR-10.ipynb
```

Once the notebook is restored, run its cells in order to load CIFAR-10, preprocess and augment images, build the CNN, train the baseline and tuned configurations, and evaluate the results. TensorFlow's CIFAR-10 loader normally downloads the dataset automatically on first use.

## Evaluation and interpretation

The documented comparison uses **test-set accuracy**. To make the project easier to audit, the restored notebook should also display training/validation learning curves and clearly identify the final test-evaluation cells. It can optionally include per-class accuracy, a confusion matrix, and examples of incorrect predictions; these additional outputs are **recommendations**, not verified existing features.

The reported accuracy improvement is encouraging, but both runs should use the same held-out test split. Hyperparameters should be selected using validation performance, and the final test set should ideally be evaluated only after selection.

## Reproducibility checklist

- [ ] Upload the complete `CIFAR-10.ipynb` notebook (currently 0 bytes).
- [ ] Verify the notebook's preprocessing, architecture, and hyperparameters match this README.
- [ ] Rerun baseline and tuned experiments with a fixed seed.
- [ ] Confirm test accuracy and checkpoint filenames from the generated results.
- [ ] Add reproducible training/evaluation outputs, if desired.

## Technologies

Python · TensorFlow · Keras · NumPy · Matplotlib · Jupyter Notebook

## Author

[MadhuRengan](https://github.com/MadhuRengan)
