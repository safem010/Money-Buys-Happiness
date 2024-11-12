# Money-Buys-Happiness
An *advanced machine learning project* for predicting stock options prices using a Transformer-LSTM hybrid model. The project preprocesses financial data, engineers relevant features, trains a model on historical data, and evaluates its performance. The project is built with PyTorch and includes modules for data processing, feature engineering, training, and evaluation.

## Table of Contents
- **Project Overview**
- **Directory Structure**
- **Getting Started**
- **Usage**
- **Configuration**
- **Feature Engineering**
- **Model Architecture**
- **Evaluation Metrics**
- **Results**
- **Contributing**
- **License**

## Project Overview
This project aims to predict option prices by leveraging a Transformer-LSTM model on engineered financial data. It takes advantage of historical data, lagged features, rolling statistics, and technical indicators to enhance model accuracy. This tool can be used for financial analysis, backtesting, and predictive analytics.

## Getting Started

### Prerequisites
- *Python 3.7+*
- *PyTorch*

Install all required Python packages by running:
```bash
pip install -r requirements.txt

```

## usage
```
The main.py file serves as the entry point for this project. It offers three modes of operation:

*Train the Model: Trains a new model on the provided dataset.*
  Select "1" when prompted.

*Load a Pre-trained Model: Loads a saved model checkpoint for further evaluation or inference.*
  Select "2" when prompted.

*Evaluate the Model: Evaluates the model on the test set using metrics and plots.*
  Select "3" when prompted.
```
## Configurtions (adjustable parameters)
```
  DATA_PATH: Path to the data file.
  SEQUENCE_LENGTH: Number of time steps in each sequence.
  TRAIN_RATIO: Proportion of data used for training.
  CHECKPOINT_PATH: File path for saving model checkpoints.
  NUM_EPOCHS: Number of training epochs.
  BATCH_SIZE: Number of samples per batch.
  LEARNING_RATE: Learning rate for the optimizer.
  LSTM_HIDDEN_DIM: Number of hidden units in the LSTM layer.
  TRANSFORMER_HEADS: Number of attention heads in the Transformer encoder.
  TRANSFORMER_LAYERS: Number of Transformer encoder layers.
  MAX_FEATURES: Maximum number of features after feature engineering.  

```
## Feature Engineering
The feature_engineering.py module adds critical features to improve model accuracy:
  Lagged Features: Adds lagged values for historical dependencies.
  Rolling Statistics: Adds moving averages and standard deviations to capture trends and volatility.
  Technical Indicators: Adds common indicators like RSI, MACD, and Bollinger Bands. If the data provides RSI, then RSI will not be used from here.


## Model Architecture
  The model combines the sequential learning capabilities of LSTMs with the attention-based Transformer layers. This architecture helps capture both short-term  dependencies and broader trends in the financial data.

Evaluation Metrics
  The following metrics are computed in evaluate.py to measure model performance:
  
  MSE (Mean Squared Error)
  MAE (Mean Absolute Error)
  R2 Score (Coefficient of Determination)
  MAPE (Mean Absolute Percentage Error)
  MSLE (Mean Squared Logarithmic Error)


## Prediction Visualization
The plot_predictions function in evaluate.py visualizes actual vs. predicted values, along with residuals, to understand the model's accuracy over time.

## Results
  Initial tests show promising results, though the model’s effectiveness will depend on data quality and feature engineering. You may run your experiments by adjusting hyperparameters in main.py and viewing evaluation plots for performance insights.

## Contributing
  - Safe Mustafa
  - Richard Corriea
  Contributions are welcome! If you'd like to add features, fix bugs, or improve documentation, please submit a pull request. 

## License
This project is open source and available under the MIT License.
    
