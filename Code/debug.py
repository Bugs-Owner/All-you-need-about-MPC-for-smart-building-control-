import torch
import torch.nn as nn
import torch.optim as optim


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, cnn_out_channels, kernel_size, lstm_layers, output_dim):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim,
                             out_channels=cnn_out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size // 2)

        self.lstm = nn.LSTM(input_size=cnn_out_channels,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

    def forward(self, x, future_steps=24):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Pass through CNN
        x = x.permute(0, 2, 1)  # [batch_size, features, seq_length]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, cnn_out_channels]

        # Initialize hidden state and cell state
        h_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(x.device)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # Initialize the output sequence
        outputs = []

        # Self-loop for future predictions
        for _ in range(future_steps):
            out = self.fc(lstm_out[:, -1, :])  # Get the last output of LSTM
            outputs.append(out.unsqueeze(1))

            # Prepare the input for the next step
            out = out.unsqueeze(2)  # Reshape for CNN input
            out = self.cnn(out).permute(0, 2, 1)  # Pass through CNN
            lstm_out, (h_n, c_n) = self.lstm(out, (h_n, c_n))  # Pass through LSTM again

        outputs = torch.cat(outputs, dim=1)  # Concatenate outputs along the time axis
        return outputs


# Example usage:
input_dim = 8
hidden_dim = 64
cnn_out_channels = 32
kernel_size = 3
lstm_layers = 2
output_dim = 1

model = CNNLSTMModel(input_dim, hidden_dim, cnn_out_channels, kernel_size, lstm_layers, output_dim)

# Input sequence with 48 steps and 8 features
input_seq = torch.randn(16, 48, 8)  # [batch_size, seq_length, features]

# Predicting the next 24 steps of solar radiation
output_seq = model(input_seq, future_steps=24)
print(output_seq.shape)  # Expected output shape: [16, 24, 1]
