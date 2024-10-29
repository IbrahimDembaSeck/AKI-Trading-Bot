import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


### 1. Weight Initialization ###

def initialize_weights(model):
    """Initialize weights for Linear and Conv2d layers using Xavier Uniform initialization."""
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


### 2. Custom Loss Functions ###

def asymmetric_loss(predictions, targets, factor=2.0):
    """Loss function that penalizes over-predictions more than under-predictions."""
    errors = predictions - targets
    loss = torch.where(errors > 0, errors ** 2, factor * (errors ** 2))
    return torch.mean(loss)


def mean_absolute_error(predictions, targets):
    """Mean Absolute Error (MAE) calculation."""
    return torch.mean(torch.abs(predictions - targets)).item()


def root_mean_squared_error(predictions, targets):
    """Root Mean Squared Error (RMSE) calculation."""
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


### 3. Learning Rate Scheduler ###

def adjust_learning_rate(optimizer, epoch, init_lr=0.01, decay_rate=0.1, decay_epoch=10):
    """Adjusts the learning rate based on the current epoch."""
    lr = init_lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


### 4. Data Normalization ###

def normalize_data(data):
    """Normalize data to a 0-1 range and return the scaler for inverse transformations."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler


def denormalize_data(data, scaler):
    """Inverse normalization using the scaler."""
    return scaler.inverse_transform(data)


### 5. Early Stopping ###

class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.counter = 0


### 6. Save and Load Model Checkpoints ###

def save_checkpoint(model, optimizer, epoch, filepath="checkpoint.pth"):
    """Save model checkpoint for a specific epoch."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint from file."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']


### 7. Gradient Clipping ###

def clip_gradients(model, clip_value=1.0):
    """Clips gradients to prevent exploding gradients, especially useful for RNNs."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)


### 8. Model Evaluation ###

def evaluate_model(model, test_loader, criterion):
    """Evaluates the model on a test set and returns the average loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(test_loader)


### 9. Save Metrics ###

def save_metrics(metrics, epoch, filepath="results/metrics"):
    """
    Save metrics (e.g., loss, MAE) as a JSON file in the metrics folder.
    Arguments:
    - metrics: dict of metric names and values.
    - epoch: current epoch number.
    - filepath: path to the metrics folder.
    """
    os.makedirs(filepath, exist_ok=True)
    file_path = os.path.join(filepath, f"metrics_epoch_{epoch}.json")
    with open(file_path, 'w') as f:
        json.dump(metrics, f)


### 10. Save Predictions ###

def save_predictions(predictions, filepath="results/predictions", filename="predictions.csv"):
    """
    Save model predictions to a CSV file.
    Arguments:
    - predictions: a dictionary or DataFrame containing predicted values.
    - filepath: path to the predictions folder.
    - filename: name of the CSV file.
    """
    os.makedirs(filepath, exist_ok=True)
    file_path = os.path.join(filepath, filename)
    if isinstance(predictions, pd.DataFrame):
        predictions.to_csv(file_path, index=False)
    else:
        pd.DataFrame(predictions).to_csv(file_path, index=False)


### 11. Save Plots ###

def save_plot(fig, filename, filepath="results/plots"):
    """
    Save a matplotlib figure to the plots folder.
    Arguments:
    - fig: the matplotlib figure to save.
    - filename: name of the plot file.
    - filepath: path to the plots folder.
    """
    os.makedirs(filepath, exist_ok=True)
    fig_path = os.path.join(filepath, filename)
    fig.savefig(fig_path)


### Additional Plotting Functions ###

def plot_metric(metric_values, metric_name="Loss", save=False, filepath="results/plots"):
    """
    Plot a given metric (e.g., loss over epochs).
    Arguments:
    - metric_values: list or array of metric values over epochs.
    - metric_name: name of the metric for labeling.
    - save: whether to save the plot to the plots folder.
    - filepath: path to save the plot if save=True.
    """
    fig, ax = plt.subplots()
    ax.plot(metric_values, label=metric_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True)

    if save:
        save_plot(fig, f"{metric_name}_plot.png", filepath)
    plt.show()
