# Republish-Predicting-Eye-Fixations
COMSM0045 Applied Deep Learning Coursework

## Predicting Eye Fixations using MrCNN/MrCNNs

This project implements a CNN to predict eye fixations based on multi-resolution input images. The implementation supports two types of models: `MrCNN` (separate branches) and `MrCNNs` (shared branches).

## Features

- **Multi-resolution Input:** The models process inputs are three samples from different resolutions 400x400, 250x250, and 150x150 respectively.
- **Two Model Architectures:** Choose between `MrCNN` or `MrCNNs` for different use cases.
- **Performance Metrics:** Calculates AUC and Shuffled AUC for evaluation.
- **Feature Visualisation:** Supports feature map visualisation during training (for the first batch).
- **Logging and Checkpointing:** Logs metrics to TensorBoard and saves model checkpoints.
- **Visualise the Feature Maps:** Visualise the feature maps of the first batch of the training data.
- **Base and Extension Support:** Run either the base implementation (`train_base.py`) or the extension (`main.py`) depending on your requirements.

## Requirements

Install the required dependencies using:
```bash
conda env create -f dl_env.yml
```

## Dataset

- **Training Dataset:** Place the training dataset at `../dataset/train_data.pth.tar`.
- **Validation Dataset:** Place the validation dataset at `../dataset/val_data.pth.tar`.
- **Validation Ground Truth:** If not already present, put `ALLFIXATIONMAPS` folder under the dataset folder. It will be automatically loaded from `../dataset/ALLFIXATIONMAPS` and saved to `../dataset/test_ground_truth` and `../dataset/val_ground_truth`.

## Usage

### Training

For the **base implementation**, use:
```bash
python train_base.py
```

For the **extension**, run the training script with the desired parameters:
```bash
python main.py --model MrCNNs --epochs 50 --batch-size 128 --learning-rate 0.001 --dropout 0.5
```

### Testing

To test a trained model, use the following command:
```bash
python test.py --model MrCNNs --model_path "./pre_trained_models/MrCNN_best.pth"
```
If you want test the MrCNN model, replace `MrCNNs` with `MrCNN`.
### Arguments for Extension

| Argument                  | Default Value          | Description                                           |
|---------------------------|------------------------|-------------------------------------------------------|
| `--model`                 | `MrCNN`               | Model type (`MrCNN` or `MrCNNs`).                    |
| `--learning-rate`         | `0.001`               | Learning rate for the optimizer.                     |
| `--batch-size`            | `128`                 | Number of images per mini-batch.                     |
| `--epochs`                | `20`                  | Number of epochs for training.                       |
| `--val-frequency`         | `1`                   | Validate the model every N epochs.                   |
| `--checkpoint-frequency`  | `5`                   | Save model checkpoints every N epochs.               |
| `-j` or `--worker-count`  | System CPU Count      | Number of worker processes for data loading.         |
| `--dropout`               | `0.5`                 | Dropout rate used in the model to prevent overfitting. |


### Logging and Visualisation

Logs are saved to the directory specified by `--log-dir`. Use TensorBoard to visualise the training process:
```bash
tensorboard --logdir=logs/
```

### Checkpoints

Model checkpoints are saved in the `./checkpoint` directory. The best model and epoch-specific models are stored.

### Validation

Validation is performed during training at the specified frequency (`--val-frequency`). AUC and Shuffled AUC metrics are calculated.

## Code Structure

- `train_base.py`: Base implementation for training.
- `main.py`: Extended training and validation script.
- `MrCNN.py`: Definition of the `MrCNN` model.
- `MrCNNs.py`: Definition of the `MrCNNs` model.
- `dataset.py`: Dataset class (`MIT`) and helper functions for loading data.
- `metrics.py`: Functions for calculating AUC and Shuffled AUC.
- `test.py`: Script for testing the model on the test set.

## Model Saving

After training, the best model is saved as `best.pth` in the checkpoint directory. The final model is also saved as `final_model.pth`.

## Testing Pre-trained Models

To test a pre-trained model, use the `test.py` script. Example:
```bash
python test.py --model_path "./pre_trained_models/MrCNN_best.pth"
```

This will load the pre-trained model from the specified path and evaluate its performance on the test dataset.
