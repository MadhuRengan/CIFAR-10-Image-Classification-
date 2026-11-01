# CIFAR-10 Image Classification (TensorFlow / Keras)

A small convolutional neural network, built from scratch (no pre-trained model), that classifies CIFAR-10 images into 10 classes. The project covers the full workflow: data pipeline, model design, training, evaluation and hyperparameter tuning.

**Result:** test accuracy improved from **85.58%** (baseline) to **87.53%** (tuned).

## Project structure

```
CIFAR-10-Image-Classification-/
├── CIFAR-10.ipynb      # full notebook: data, model, training, evaluation, tuning
├── requirements.txt    # Python dependencies
└── README.md
```

Files created when the notebook runs (not stored in the repo): `baseline_best.keras` and `tuned_best.keras` (best checkpoints).

## How to run

```bash
git clone https://github.com/MadhuRengan/CIFAR-10-Image-Classification-.git
cd CIFAR-10-Image-Classification-

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook CIFAR-10.ipynb   # then run the cells from top to bottom
```

The dataset is downloaded automatically by `tensorflow.keras.datasets.cifar10` the first time the notebook runs.
Training on a CPU takes roughly 1-2 minutes per epoch. A GPU (for example on Google Colab) is much faster.

## Training setup

| Item | Setting |
|---|---|
| Data split | 45,000 train / 5,000 validation / 10,000 test |
| Normalization | per-channel mean/std, computed on the training set only |
| Augmentation | random horizontal flip + random crop (4-pixel padding), training data only |
| Data loading | `tf.data` pipeline: shuffle, parallel map, batch, prefetch |
| Loss | sparse categorical cross-entropy |
| Metric | accuracy |
| Optimizer | Adam |
| LR schedule | `ReduceLROnPlateau` (halve the learning rate after 3 epochs without validation-loss improvement) |
| Checkpoints | best model by validation accuracy, saved with `ModelCheckpoint` |
| Epochs | 30 |
| Reproducibility | `tf.keras.utils.set_random_seed(42)` |

## Model design decisions

The model has about 290K parameters: three stages with 32, 64 and 128 filters.

- **Conv + BatchNorm + ReLU blocks:** convolutions detect patterns. BatchNorm keeps activations stable, which makes training faster and steadier. `use_bias=False` is used because BatchNorm already has its own shift.
- **Two blocks per stage, then MaxPooling:** pooling halves the image size (32 → 16 → 8 → 4) and cuts computation.
- **Dropout:** randomly switches off neurons during training to reduce overfitting.
- **Global average pooling + Dense(10, softmax):** the classification head has far fewer parameters than flattening everything into a dense layer.

## Evaluation

The notebook plots the loss and accuracy curves for the training and validation sets, reports the test accuracy, and shows a grid of misclassified test images.

## Hyperparameter tuning

I tuned four hyperparameters, changing **one at a time** from the baseline (learning rate 1e-3, batch size 128, Adam, dropout 0.3). Each experiment used the same random seed and trained for 5 epochs, and was compared on validation accuracy. The test set was used only at the end.

| Setting changed | Value tried | 5-epoch validation accuracy |
|---|---|---|
| Baseline | lr 1e-3, batch 128, Adam, dropout 0.3 | 0.6846 |
| Learning rate | 1e-2 | 0.6420 |
| Learning rate | 1e-4 | 0.4986 |
| Batch size | 64 | 0.7128 |
| Batch size | 256 | 0.6434 |
| Optimizer | SGD + momentum (lr 1e-2) | 0.6158 |
| Optimizer | RMSprop | 0.6644 |
| Dropout | 0.1 | 0.7456 |
| Dropout | 0.5 | 0.6354 |

The best value of each setting (learning rate 1e-3, batch size 64, Adam, dropout 0.1) was combined and retrained for 30 epochs.

### Baseline vs tuned (30 epochs)

| Model | Batch size | Dropout | Optimizer | Learning rate | Test accuracy |
|---|---|---|---|---|---|
| Baseline | 128 | 0.3 | Adam | 1e-3 | 85.58% |
| Tuned | 64 | 0.1 | Adam | 1e-3 | **87.53%** |

### Observations and limitations

- A learning rate of 1e-3 worked best. 1e-4 learned too slowly in 5 epochs, and 1e-2 was less stable.
- Smaller batches did better in short runs because they make more weight updates per epoch.
- Lower dropout looked best partly because 5 epochs is too short for overfitting to appear. The 30-epoch run shows a small gap between training and validation loss, which is the cost of lower dropout.
- Only one setting was changed at a time, so combinations were not searched. Each result is from a single run, so differences of 1-2% in the short experiments may be noise.
