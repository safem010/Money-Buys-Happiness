import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, \
    mean_absolute_percentage_error
import numpy as np
from tqdm import tqdm


def evaluate_model(model, test_loader, criterion, device=torch.device("cpu")):
    """
    Evaluates model performance on the test set with extended metrics and logging.
    Arguments:
    - model: The trained model to evaluate.
    - test_loader: DataLoader for the test set.
    - criterion: Loss function for computing the test loss.
    - device: Device to run the evaluation on (default is CPU).

    Returns:
    - metrics: A dictionary containing MSE, MAE, R², MAPE, MSLE, and average loss on the test set.
    - actual: List of actual target values from the test set.
    - predicted: List of predicted target values from the model.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    actual = []
    predicted = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Get predictions
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # Collect actual and predicted values for metric calculation
            actual.extend(labels.cpu().numpy())
            predicted.extend(outputs.cpu().numpy())

    # Calculate mean loss over all test samples
    test_loss /= len(test_loader.dataset)

    # Calculate regression metrics
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    msle = mean_squared_log_error(actual, predicted)

    # Calculate prediction confidence intervals (std deviation)
    prediction_std_dev = np.std(predicted)

    # Output results in a dictionary
    metrics = {
        "Test Loss": test_loss,
        "MSE": mse,
        "MAE": mae,
        "R2 Score": r2,
        "MAPE": mape,
        "MSLE": msle,
        "Prediction Std Dev": prediction_std_dev
    }

    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics, actual, predicted


def plot_predictions(actual, predicted, title="Model Predictions vs Actual", save_path=None, plot_residuals=False):
    """
    Plots the actual vs predicted values and optional residuals for the test set.
    Arguments:
    - actual: List or array of actual target values.
    - predicted: List or array of predicted target values.
    - title: Title of the plot.
    - save_path: If provided, saves the plot to the specified path.
    - plot_residuals: If True, plots residuals as an additional subplot.
    """
    plt.figure(figsize=(12, 8))

    # Main Prediction Plot
    plt.subplot(2, 1, 1)
    plt.plot(actual, label="Actual", color="blue")
    plt.plot(predicted, label="Predicted", color="red", linestyle="--")
    plt.xlabel("Samples")
    plt.ylabel("Target Value")
    plt.title(title)
    plt.legend()

    # Residuals Plot
    if plot_residuals:
        residuals = np.array(actual) - np.array(predicted)
        plt.subplot(2, 1, 2)
        plt.plot(residuals, label="Residuals", color="purple")
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel("Samples")
        plt.ylabel("Residuals")
        plt.title("Residuals (Actual - Predicted)")
        plt.legend()

    # Save or show plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
