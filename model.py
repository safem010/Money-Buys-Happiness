import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Positional encoding for odd `d_model` (input_dim)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:  # Adjust for odd input_dim
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerLSTMModel(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, transformer_heads=8, transformer_layers=6, lstm_layers=6,
                 max_len=5000, dropout=0.3):
        super(TransformerLSTMModel, self).__init__()

        # Dynamically adjust transformer_heads for input_dim compatibility
        if input_dim % transformer_heads != 0:
            original_heads = transformer_heads
            transformer_heads = max(1, input_dim // (input_dim // transformer_heads))
            print(f"Adjusted transformer_heads from {original_heads} to {transformer_heads} for compatibility.")

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=input_dim, max_len=max_len)

        # Transformer Encoder Stack 1
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=input_dim, nhead=transformer_heads, dropout=dropout)
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=transformer_layers)
        self.layer_norm1 = nn.LayerNorm(input_dim)

        # Transformer Encoder Stack 2 (Optional)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=input_dim, nhead=transformer_heads, dropout=dropout)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=transformer_layers)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        # LSTM Stack
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout)
        self.layer_norm3 = nn.LayerNorm(lstm_hidden_dim)

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single output for regression
        )

    def forward(self, x):
        # Positional Encoding
        x = self.positional_encoding(x)

        # Transformer Encoding Stack 1
        x = x.permute(1, 0, 2)  # Transform to (seq_len, batch, input_dim) for transformer
        x = self.transformer_encoder1(x)
        x = self.layer_norm1(x)

        # Transformer Encoding Stack 2 (optional second transformer layer)
        x = self.transformer_encoder2(x)
        x = self.layer_norm2(x)

        # LSTM Processing, normalize last hidden state
        x = x.permute(1, 0, 2)  # Transform back to (batch, seq_len, input_dim) for LSTM
        x, (h_n, c_n) = self.lstm(x)
        x = self.layer_norm3(h_n[-1])  # Normalize last LSTM hidden state

        # Fully Connected Layers with Dropout and ReLU activations
        output = self.fc_layers(x)
        return output
