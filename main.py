import torch
import torch.nn as nn
from data_processing.data_loader import load_data, clean_data, create_sequences, split_data
from data_processing.feature_engineering import FeatureEngineer
from data_processing.dataset import SequenceDataset
from model.model import TransformerLSTMModel
from model.train import TrainPipeline, CheckpointHandler

# Define file path and constants for configuration
DATA_PATH = "data_processing/data.xlsx"
SEQUENCE_LENGTH = 30
TRAIN_RATIO = 0.7
CHECKPOINT_PATH = "saved_model_checkpoint.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01
LSTM_HIDDEN_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
MAX_FEATURES = 11

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data():
    print("Module 1: Loading and Preprocessing Data")
    data = load_data(DATA_PATH)
    data = clean_data(data)
    print("Data after cleaning and scaling:\n", data.head())

    print("Module 2: Applying Feature Engineering")
    feature_engineer = FeatureEngineer(data, max_features=MAX_FEATURES)  # Use MAX_FEATURES from config
    data = feature_engineer.run_feature_engineering()
    print("Data after feature engineering:\n", data.head())
    print(f"Final input dimension (feature count): {data.shape[1]}")

    # Split into sequences, etc.
    sequences, labels = create_sequences(data, SEQUENCE_LENGTH)
    train_data, val_data, test_data = split_data(sequences, labels, TRAIN_RATIO)
    print(f"Training data size: {len(train_data[0])}")
    print(f"Validation data size: {len(val_data[0])}")
    print(f"Test data size: {len(test_data[0])}")
    return train_data, val_data, test_data, data.shape[1]

# Module 2: Training function
def train():
    train_data, val_data, _, input_dim = preprocess_data()

    # Logging to ensure transformer_heads is compatible with input_dim
    if input_dim % TRANSFORMER_HEADS != 0:
        print(f"Warning: input_dim ({input_dim}) is not divisible by transformer_heads ({TRANSFORMER_HEADS}).")
        print("Consider adjusting transformer_heads or input_dim to avoid AssertionError.")

    # Create DataLoaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(SequenceDataset(*train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(SequenceDataset(*val_data), batch_size=BATCH_SIZE)

    # Initialize the model with logging
    print(f"Initializing model with input_dim={input_dim}, lstm_hidden_dim={LSTM_HIDDEN_DIM}, "
          f"transformer_heads={TRANSFORMER_HEADS}, transformer_layers={TRANSFORMER_LAYERS}")
    model = TransformerLSTMModel(
        input_dim=input_dim,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_layers=TRANSFORMER_LAYERS
    ).to(device)

    # Initialize and start the training pipeline
    pipeline = TrainPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_path=CHECKPOINT_PATH
    )

    pipeline.train_model()

# Module 3: Load pre-trained model
def load():
    print("Loading mode selected.")
    _, _, _, input_dim = preprocess_data()

    # Logging input_dim and transformer_heads compatibility check
    if input_dim % TRANSFORMER_HEADS != 0:
        print(f"Warning: input_dim ({input_dim}) is not divisible by transformer_heads ({TRANSFORMER_HEADS}).")

    # Initialize the model
    print(f"Initializing model with input_dim={input_dim}, lstm_hidden_dim={LSTM_HIDDEN_DIM}, "
          f"transformer_heads={TRANSFORMER_HEADS}, transformer_layers={TRANSFORMER_LAYERS}")
    model = TransformerLSTMModel(
        input_dim=input_dim,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_layers=TRANSFORMER_LAYERS
    ).to(device)

    # Load the model checkpoint
    CheckpointHandler.load_checkpoint(model, CHECKPOINT_PATH, device)
    print("Model loaded successfully.")

# Module 4: Evaluation function
def evaluate():
    print("Evaluation mode selected.")
    _, _, test_data, input_dim = preprocess_data()

    # Logging input_dim and transformer_heads compatibility check
    if input_dim % TRANSFORMER_HEADS != 0:
        print(f"Warning: input_dim ({input_dim}) is not divisible by transformer_heads ({TRANSFORMER_HEADS}).")

    # Create DataLoader for the test set
    test_loader = torch.utils.data.DataLoader(SequenceDataset(*test_data), batch_size=BATCH_SIZE)

    # Initialize the model
    print(f"Initializing model with input_dim={input_dim}, lstm_hidden_dim={LSTM_HIDDEN_DIM}, "
          f"transformer_heads={TRANSFORMER_HEADS}, transformer_layers={TRANSFORMER_LAYERS}")
    model = TransformerLSTMModel(
        input_dim=input_dim,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_layers=TRANSFORMER_LAYERS
    ).to(device)

    # Load the model checkpoint for evaluation
    CheckpointHandler.load_checkpoint(model, CHECKPOINT_PATH, device)

# Module 5: Main function with user-friendly menu
def main():
    print("Please select an operation mode:")
    print("1. Train the model")
    print("2. Load a pre-trained model")
    print("3. Evaluate the model")

    choice = input("Enter the number of your choice (1, 2, or 3): ")

    if choice == "1":
        train()
    elif choice == "2":
        load()
    elif choice == "3":
        evaluate()
    else:
        print("Invalid choice. Please run the program again and select a valid option.")

    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
