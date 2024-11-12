import torch
import torch.nn as nn
from tqdm import tqdm

# submodule 1
class TrainPipeline:
    def __init__(self, model, train_loader, val_loader, device, num_epochs, learning_rate, checkpoint_path):
        """
        Initializes the training pipeline with model, dataloaders, and training parameters.
        Arguments:
        - model: The model to be trained
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - device: Device to run the training on (e.g., "cuda" or "cpu")
        - num_epochs: Number of training epochs
        - learning_rate: Learning rate for optimizer
        - checkpoint_path: Path to save the model checkpoint
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path

        # Define training components
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=4)
        self.scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
        print(f"Using device: {device}")

    def train_model(self):
        """
        Trains the model using the provided dataloaders and parameters.
        Implements early stopping based on validation loss.
        """
        best_val_loss = float('inf')
        early_stopping_patience = 4
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}/{self.num_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()

                # Gradient clipping and step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item() * inputs.size(0)

            # Compute and print training loss
            train_loss = running_loss / len(self.train_loader.dataset)
            print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")
            self.scheduler.step(train_loss)

            # Validation step
            if self.val_loader:
                val_loss = self.validate_model()
                if val_loss < best_val_loss:
                    print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                    CheckpointHandler.save_model(self.model, self.checkpoint_path)
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print("Early stopping triggered due to no improvement.")
                        break

        return self.model

    def validate_model(self):
        """
        Validates the model on the validation set.
        Returns:
        - val_loss: Validation loss over the entire validation set.
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(self.val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")
        return val_loss


# submodule 2: change this to a util class with saving and loading checkpoints
class CheckpointHandler:
    @staticmethod
    def save_model(model, path):
        """
        Saves the model's state dictionary to the specified path.
        Arguments:
        - model: The model to save.
        - path: The file path to save the model.
        """
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    @staticmethod
    def load_checkpoint(model, path, device):
        """
        Loads model weights from a checkpoint.
        Arguments:
        - model: The model to load weights into.
        - path: Path to the checkpoint file.
        - device: The device to map the model to after loading weights.

        Returns:
        - model: The model with loaded weights.
        """
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
        return model.to(device)
