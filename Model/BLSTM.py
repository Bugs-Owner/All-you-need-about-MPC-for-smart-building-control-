import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# class BayesianLSTM(nn.Module):
#
#     def __init__(self, par, device):
#         super(BayesianLSTM, self).__init__()
#
#         self.batch_size = par["training_batch"]
#         self.hidden_size_1 = par["Encoder_hidd"]
#         self.hidden_size_2 = par["Decoder_hidd"]
#         self.device = device
#         self.stacked_layers = 2
#         self.dropout_probability = 0.2
#
#         self.lstm1 = nn.LSTM(par["BLSTM_input"],
#                              self.hidden_size_1,
#                              num_layers=self.stacked_layers,
#                              batch_first=True)
#         self.lstm2 = nn.LSTM(self.hidden_size_1,
#                              self.hidden_size_2,
#                              num_layers=self.stacked_layers,
#                              batch_first=True)
#
#         self.fc = nn.Linear(self.hidden_size_2, par["BLSTM_output"])
#
#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         hidden = self.init_hidden1(batch_size)
#         output, _ = self.lstm1(x, hidden)
#         output = F.dropout(output, p=self.dropout_probability, training=True)
#         state = self.init_hidden2(batch_size)
#         output, state = self.lstm2(output, state)
#         output = F.dropout(output, p=self.dropout_probability, training=True)
#         output = output[:, -1, :]
#         y_pred = self.fc(output)
#         return y_pred
#
#     def init_hidden1(self, batch_size):
#         hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
#         cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
#         return hidden_state.to(self.device), cell_state.to(self.device)
#
#     def init_hidden2(self, batch_size):
#         hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
#         cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
#         return hidden_state.to(self.device), cell_state.to(self.device)

class BayesianLSTM(nn.Module):

    def __init__(self, par, device):
        super(BayesianLSTM, self).__init__()
        self.par = par
        self.batch_size = par["training_batch"]
        self.hidden_size_1 = par["Encoder_hidd"]
        self.hidden_size_2 = par["Decoder_hidd"]
        self.device = device
        self.stacked_layers = 2
        self.dropout_probability = 0.2

        self.cnn = nn.Sequential(
                        nn.Conv1d(in_channels=par["CNN_in"], out_channels=par["CNN_out"], kernel_size=par["CNN_kernel_size"], stride=par["CNN_stride"], padding=par["CNN_padding"]),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=par["Pool_kernel_size"], stride=par["Pool_stride"]),
                    )

        self.lstm1 = nn.LSTM(par["BLSTM_input"],
                             self.hidden_size_1,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)

        self.fc = nn.Linear(self.hidden_size_2, 1)

    def forward(self, x):
        x_encoder = x[:,:self.par['win_in'],:]
        x_decoder = x[:,self.par['win_in']:,:]
        batch_size, seq_len, _ = x_encoder.size()
        x_encoder = x_encoder.permute(0, 2, 1)
        out = self.cnn(x_encoder)
        out = out.permute(0, 2, 1)

        hidden = self.init_hidden1(batch_size)

        # state = self.init_hidden2(batch_size)
        # output, state = self.lstm2(output, state)
        # output = F.dropout(output, p=self.dropout_probability, training=True)
        out_list = []
        for t in range(self.par['win_out']):
            if t > 0:
                out = y_pred.unsqueeze(1)
            output, hidden = self.lstm1(out, hidden)
            output = F.dropout(output, p=self.dropout_probability, training=True)
            output = output[:, -1, :]
            y_pred = self.fc(output)
            out_list.append(y_pred)

        outputs = torch.cat(out_list, dim=1)
        return outputs

    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        return hidden_state.to(self.device), cell_state.to(self.device)

    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        return hidden_state.to(self.device), cell_state.to(self.device)