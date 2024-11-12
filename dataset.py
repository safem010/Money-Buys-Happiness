from torch.utils.data import Dataset
import torch


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = [torch.tensor(label, dtype=torch.float32) for label in labels]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Ensure sequence has the correct shape
        assert sequence.shape[0] > 0, f"Found empty sequence at index {idx}"
        return sequence, label
