import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# data path: data_processing/data.xlsx

def load_data(filepath):
    """
    step 1: load data
    - reads the dataset from an Excel file
    - returns a DataFrame with the raw data
    """
    data = pd.read_excel(filepath)
    return data

def clean_data(data):
    """
    step 2: clean data
    - removes missing values, normalizes numerical columns individually
    - uses MinMaxScaler for volume and StandardScaler for other features
    """
    data = data.dropna()  # drop rows with missing values

    # scale volume columns with MinMaxScaler (0 to 1 range)
    volume_cols = ['OBV_x', 'OBV_y']
    min_max_scaler = MinMaxScaler()
    data[volume_cols] = min_max_scaler.fit_transform(data[volume_cols])

    # scale other numerical columns with StandardScaler (mean 0, std 1)
    other_numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.difference(volume_cols)
    standard_scaler = StandardScaler()
    data[other_numerical_cols] = standard_scaler.fit_transform(data[other_numerical_cols])

    return data


def create_sequences(data, sequence_length):
    """
    step 3: create sequences
    - generates sequences of length 'sequence_length' for time-series modeling
    - each sequence has a corresponding target label from 'option_change_target'
    - returns lists of sequences and labels, with each sequence having shape (sequence_length, number_of_features)
    """
    sequences, labels = [], []

    for i in range(len(data) - sequence_length):
        sequence = data.iloc[i:i + sequence_length].values
        label = data['option_change_target'].iloc[i + sequence_length]

        # Check for correct shape before appending
        if sequence.shape == (sequence_length, data.shape[1]):
            sequences.append(sequence)
            labels.append(label)
        else:
            print(f"Skipping sequence at index {i} due to shape mismatch: {sequence.shape}")

    print(f"Generated {len(sequences)} sequences with length {sequence_length}")
    return sequences, labels



def split_data(sequences, labels, train_ratio=0.7):
    """
    step 4: split data
    - splits sequences and labels into train, validation, and test sets
    """
    train_size = int(len(sequences) * train_ratio)
    val_size = int(len(sequences) * ((1 - train_ratio) / 2))

    # training data
    train_sequences = sequences[:train_size]
    train_labels = labels[:train_size]

    # validation data
    val_sequences = sequences[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]

    # test data
    test_sequences = sequences[train_size + val_size:]
    test_labels = labels[train_size + val_size:]

    return (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels)


def test_data_pipeline(filepath, seq_length=30, train_ratio=0.7):
    """
    step 5: test data pipeline
    - loads data, applies cleaning and scaling, creates sequences, and splits data
    - prints data samples and structure at each step for verification
    """
    # 5a) Load data
    print("Loading data...")
    data = load_data(filepath)
    print("Initial data sample:\n", data.head())
    print("Data shape:", data.shape)

    # 5b) Clean data
    print("\nCleaning and scaling data...")
    data = clean_data(data)
    print("Sample after cleaning and scaling:\n", data.head())
    print("Data shape after cleaning:", data.shape)

    # 5c) Create sequences
    print("\nCreating sequences...")
    sequences, labels = create_sequences(data, seq_length)
    print(f"Number of sequences created: {len(sequences)}")
    print("Sample sequence shape:", sequences[0].shape if sequences else "N/A")
    print("Sample sequence (first 2 time steps):\n", sequences[0][:2] if sequences else "N/A")
    print("Sample label:", labels[0] if labels else "N/A")

    # 5d) Split data
    print("\nSplitting data into train, validation, and test sets...")
    (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels) = split_data(sequences, labels, train_ratio)

    # Print dataset sizes
    print("\nDataset sizes:")
    print("Training set:", len(train_sequences), "sequences")
    print("Validation set:", len(val_sequences), "sequences")
    print("Test set:", len(test_sequences), "sequences")

    # Final check of shapes
    print("\nFinal check of data shapes:")
    print("Train sequence shape:", train_sequences[0].shape if train_sequences else "N/A")
    print("Validation sequence shape:", val_sequences[0].shape if val_sequences else "N/A")
    print("Test sequence shape:", test_sequences[0].shape if test_sequences else "N/A")
