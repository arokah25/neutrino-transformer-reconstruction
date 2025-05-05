# Neutrino Transformer Reconstruction

This project implements a Transformer-based model to reconstruct the (x, y) position of a neutrino interaction from simulated Cherenkov photon detection data. The data is inspired by the IceCube neutrino detector and consists of variable-length sequences of photon hits, each described by detection time and detector position.

## Overview

Each event represents a point-like neutrino interaction that emits Cherenkov light, recorded by a subset of detector modules arranged in a 2D grid. The model processes the variable-length sequence of photon hits using a Transformer encoder with masking and padding to produce a fixed-size embedding for regression. The final output is the predicted interaction position in 2D space.

## Dataset

The dataset is provided in Parquet format and consists of:
- `train.pq`: Training set (approximately 200k events)
- `val.pq`: Validation set (approximately 10k events)

Each event includes:
- `data`: A NumPy array of shape (3, n_hits), representing [time, x, y] values
- `xpos`, `ypos`: Ground truth coordinates of the neutrino interaction

## Model

The Transformer architecture includes:
- Linear input projection from 3 input features to a hidden dimension
- Multiple Transformer encoder layers (at least two), with multi-head self-attention
- Padding and masking support for variable-length sequences
- Mean pooling over valid sequence elements
- Output projection to 2D space for regression

The model is implemented in PyTorch and trained using mean squared error loss.

## Evaluation

The model is evaluated using the following metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Average Euclidean Distance between predicted and true interaction positions

Visualizations include:
- Scatter plot of predicted vs. true positions
- Histogram of Euclidean prediction errors

## Results

Example performance (on validation set):
- MSE: 2.47
- MAE: 0.99
- Average Euclidean Distance: 1.57 meters

This indicates that, on average, the predicted interaction is within one detector module spacing of the true location.

## Requirements

- Python 3.8+
- PyTorch
- pandas
- pyarrow
- matplotlib
- tensorboard

Install dependencies (in Colab or locally):

```bash
pip install torch pandas pyarrow matplotlib tensorboard
```

## Usage

To run the training and evaluation pipeline:

1. Upload the `train.pq` and `val.pq` files to your Google Drive.
2. Set the `DATA_PATH` in the notebook to point to the correct directory.
3. Run the notebook `transformer_pred.ipynb` step by step.
4. Use TensorBoard to monitor training logs if desired.

## Acknowledgments

This project was completed as part of an advanced deep learning assignment involving physical event reconstruction using Transformer architectures. The dataset and task are inspired by the IceCube Neutrino Observatory.

## License

This repository is for academic and non-commercial use only.
