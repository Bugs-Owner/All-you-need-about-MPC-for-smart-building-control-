import torch
import torch.nn as nn


# class CNN_LSTM(nn.Module):
#     def __init__(self, par):
#         super(CNN_LSTM, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv1d(in_channels=par["CNN_in"], out_channels=par["CNN_out"], kernel_size=par["CNN_kernel_size"], stride=par["CNN_stride"], padding=par["CNN_padding"]),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=par["Pool_kernel_size"], stride=par["Pool_stride"]),
#         )
#         self.lstm = nn.LSTM(input_size=par["LSTM_in"], hidden_size=par["LSTM_hidd"], num_layers=par["num_layers"], batch_first=True)
#         self.fc = nn.Linear(par["FC_in"], par["FC_out"])
#
#     def forward(self, x):
#         #cnn takes input of shape (batch_size, channels, seq_len)
#         x = x.permute(0, 2, 1)
#         out = self.cnn(x)
#         # lstm takes input of shape (batch_size, seq_len, input_size)
#         out = out.permute(0, 2, 1)
#         out, _ = self.lstm(out)
#         out = self.fc(out[:, -1, :])
#         return out

class CNN_LSTM(nn.Module):
    def __init__(self, par):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=par["CNN_in"], out_channels=par["CNN_out"], kernel_size=par["CNN_kernel_size"], stride=par["CNN_stride"], padding=par["CNN_padding"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=par["Pool_kernel_size"], stride=par["Pool_stride"]),
        )
        self.lstm = nn.LSTM(input_size=par["LSTM_in"], hidden_size=par["LSTM_hidd"], num_layers=par["num_layers"], batch_first=True)
        self.fc = nn.Linear(par["FC_in"], par["FC_out"]*2)

    def forward(self, x):
        #cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        mean, log_sigma = torch.chunk(out, 2, dim=-1)
        mean = torch.exp(mean)
        sigma = torch.exp(log_sigma)
        return mean, sigma