import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_channels, cnn_kernel_size, lstm_hidden_dim, lstm_num_layers):
        super(EncoderCNNLSTM, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=cnn_kernel_size,
                             padding=cnn_kernel_size // 2)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_dim, lstm_num_layers, batch_first=True)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.dropout_probability = 0.2

    def forward(self, x):
        # CNN expects input of shape (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.lstm_num_layers, cnn_out.size(0), self.lstm_hidden_dim).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, cnn_out.size(0), self.lstm_hidden_dim).to(x.device)

        # Pass the CNN output through the LSTM layer
        lstm_out, (hn, cn) = self.lstm(cnn_out, (h0, c0))

        return hn, cn


# Define the Decoder
class DecoderLSTM(nn.Module):
    def __init__(self, decoder_input_dim, lstm_hidden_dim, lstm_num_layers, output_dim):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(decoder_input_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.dropout_probability = 0.05

    def forward(self, x, hidden, cell):
        # Pass input through the LSTM layer
        output, (hn, cn) = self.lstm(x, (hidden, cell))
        # Pass the output of the LSTM through the fully connected layer
        output = self.fc(output)
        output = F.dropout(output, p=self.dropout_probability, training=True)

        return output, hn, cn


# Define the full Encoder-Decoder Model
class EncoderDecoderCNNLSTM(nn.Module):
    def __init__(self, par):
        super(EncoderDecoderCNNLSTM, self).__init__()
        self.par = par
        self.encoder = EncoderCNNLSTM(input_dim=par["input_dim"], cnn_channels=par["cnn_channels"], cnn_kernel_size=par["cnn_kernel_size"], lstm_hidden_dim=par["lstm_encoder"], lstm_num_layers=par["lstm_num_layers"])
        self.decoder = DecoderLSTM(decoder_input_dim=par["decoder_input_dim"], lstm_hidden_dim=par["lstm_decoder"], lstm_num_layers=par["lstm_num_layers"], output_dim=1)
        self.forecast_len = self.par['win_out']

    def forward(self, x):
        x_encoder = x[:, :self.par['win_in'], :]
        x_decoder = x[:, self.par['win_in']-1:-1, :]
        # Encode the input sequence
        hidden, cell = self.encoder(x_encoder)

        # Initialize decoder input (start with the last time step of the input sequence)
        decoder_input = x_decoder[:, [0], :]  # (batch_size, 1, input_dim)
        decoder_outputs = []

        # Decode step-by-step
        for t in range(x_decoder.shape[1]):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            decoder_outputs.append(output.squeeze())
            if t < x_decoder.shape[1]-1:
                decoder_input = x_decoder[:, [t+1], :]
                decoder_input[:, :, [0]]=output
            else:
                0

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs