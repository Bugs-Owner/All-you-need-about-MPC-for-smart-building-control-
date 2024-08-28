import torch
from tqdm import trange
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='../Checkpoint', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        mdl_name = 'vali.pt'
        savemodel = os.path.join(self.path, mdl_name)
        torch.save(model.state_dict(), savemodel)
        self.val_loss_min = val_loss

def nll_loss(mean, sigma, target):
    dist = torch.distributions.Normal(mean, sigma)
    loss = -dist.log_prob(target)
    return loss.mean()

def train_model(model, train_loader, valid_loader, test_loader, lr, epochs,
                patience, scaler, win_out, rawdf, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    train_losses = []
    valid_losses = []

    with (trange(epochs) as tr):
        for epoch in tr:
            # Training
            model.train()
            criterion = nn.MSELoss()
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.squeeze())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item() * data.size(0)

            # Validation
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target.squeeze())
                    res = 50
                    valid_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if epoch % res == 0:
                device = device
                with torch.no_grad():
                    for data, target in test_loader:
                        data, _ = data.to(device), target.to(device)

                        sample={}
                        for s in range (300):
                            output = model(data).cpu()
                            outputs_denorm = []
                            for idx in range(output.shape[0]):
                                outputs_denorm.append(scaler.inverse_transform(output[[idx], :].reshape(-1, 1)))
                            sample[s]=outputs_denorm

                        #     #output = np.clip(output, 0, None)

                test_len = len(outputs_denorm)
                pred_len = win_out
                display = True
                if display == True :
                    fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.8), dpi=300, constrained_layout=True)
                    ax.plot_date(rawdf.index[:test_len], rawdf.iloc[:,[1]].values[:test_len],
                                 '-', linewidth=1, color="#159A9C", label='Groundtruth')
                    timestep = 0
                    y_true = (rawdf.iloc[:, [1]].values[:win_out][:test_len]).reshape(-1, 1)
                    mae = 0
                    columns = []
                    for s in range (300):
                        tem = sample[s][timestep][:test_len - timestep]
                        y_pred = tem
                        mae += mean_absolute_error(y_true, y_pred)
                        columns.append(pd.Series(y_pred.reshape(-1,), name='out_{}'.format(s)))
                        # ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                        #                  tem, '--', linewidth=0.1, color="gray")
                    test_df = pd.concat(columns, axis=1)
                    mean = test_df.mean(axis=1)
                    std = test_df.std(axis=1)
                    ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                                 mean, '-', linewidth=0.5, color="red", label='CNN-LSTM')
                    ax.fill_between(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                                    mean - 2*std, mean + 2*std, color ='hotpink', alpha = 0.3)
                    text_gray = 'Epochs:{}\nMAE:{:.2f}'.format(epoch, mae/(s+1))
                    ax.text(0.05, 0.8, text_gray, fontsize=6, color='gray', fontweight='bold', transform=ax.transAxes)
                    ax.tick_params(axis='both', which='minor', labelsize=7)
                    ax.tick_params(axis='both', which='major', labelsize=7)
                    ax.xaxis.set_minor_locator(dates.HourLocator(byhour=range(2, 24, 4)))
                    ax.xaxis.set_minor_formatter(dates.DateFormatter("%H"))
                    ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
                    ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
                    ax.set_xlabel(None)
                    ax.set_ylabel('Target', fontsize=7)
                    ax.margins(x=0)
                    ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=6, frameon=False)
                    plt.show()
                else:
                    0

            tr.set_postfix(epoch="{0:.0f}".format(epoch+1), train_loss="{0:.6f}".format(train_loss),
                           valid_loss="{0:.6f}".format(valid_loss))

            # Early Stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        mdl_name = 'vali.pt'
        savemodel = os.path.join('../Checkpoint', mdl_name)
        model.load_state_dict(torch.load(savemodel))
        loss_dic = {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
        }
        return model, loss_dic


# def train_model(model, train_loader, valid_loader, test_loader, lr, epochs,
#                 patience, scaler, win_out, rawdf, device):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#     early_stopping = EarlyStopping(patience=patience, verbose=True)
#     train_losses = []
#     valid_losses = []
#
#     with (trange(epochs) as tr):
#         for epoch in tr:
#             # Training
#             model.train()
#             train_loss = 0.0
#             for data, target in train_loader:
#                 data, target = data.to(device), target.to(device)
#                 optimizer.zero_grad()
#                 output = model(data)
#                 loss = criterion(output, target.squeeze())
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item() * data.size(0)
#
#             # Validation
#             model.eval()
#             valid_loss = 0.0
#             with torch.no_grad():
#                 for data, target in valid_loader:
#                     data, target = data.to(device), target.to(device)
#                     output = model(data)
#                     res = 10
#                     loss = criterion(output, target.squeeze())
#                     valid_loss += loss.item() * data.size(0)
#
#             train_loss = train_loss / len(train_loader.dataset)
#             valid_loss = valid_loss / len(valid_loader.dataset)
#             train_losses.append(train_loss)
#             valid_losses.append(valid_loss)
#
#             if epoch % res == 0:
#                 device = device
#                 with torch.no_grad():
#                     for data, target in test_loader:
#                         data, target = data.to(device), target.to(device)
#                         output = model(data)
#
#                 outputs_ = output.cpu()
#                 outputs_denorm = []
#
#                 for idx in range(outputs_.shape[0]):
#                     outputs_denorm.append(
#                         scaler.inverse_transform(outputs_[[idx], :].reshape(-1, 1)))
#
#                 test_len = len(outputs_denorm)
#                 pred_len = win_out
#                 display = True
#                 if display == True :
#                     fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.8), dpi=300, constrained_layout=True)
#                     ax.plot_date(rawdf.index[:test_len], rawdf.iloc[:,[1]].values[:test_len],
#                                  '-', linewidth=1, color="#159A9C", label='Groundtruth')
#                     timestep = 0
#                     tem = outputs_denorm[timestep][:test_len - timestep]
#                     tem = tem
#                     ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
#                                      tem, '-', linewidth=1, color="gray", label='CNN-LSTM')
#
#                     y_true = (rawdf.iloc[:,[1]].values[:win_out][:test_len]).reshape(-1, 1)
#                     y_pred = tem
#                     mae = mean_absolute_error(y_true, y_pred)
#                     mape = mean_absolute_percentage_error(y_true, y_pred) * 100
#                     #text_gray = 'Epochs:{}\nMAE:{:.2f}[°C]\nMAPE:{:.2f}[%]'.format(epoch, mae, mape)
#                     text_gray = 'Epochs:{}\nMAE:{:.2f}'.format(epoch, mae)
#                     # Add the gray text
#                     ax.text(1.01, 0.55, text_gray, fontsize=6, color='gray', fontweight='bold', transform=ax.transAxes)
#                     ax.tick_params(axis='both', which='minor', labelsize=7)
#                     ax.tick_params(axis='both', which='major', labelsize=7)
#                     ax.xaxis.set_minor_locator(dates.HourLocator(interval=4))
#                     ax.xaxis.set_minor_formatter(dates.DateFormatter("%H"))
#                     ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
#                     ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
#                     ax.set_xlabel(None)
#                     ax.set_ylabel('Target', fontsize=7)
#                     ax.margins(x=0)
#                     ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=6, frameon=False)
#                     plt.show()
#                 else:
#                     0
#
#             tr.set_postfix(epoch="{0:.0f}".format(epoch+1), train_loss="{0:.6f}".format(train_loss),
#                            valid_loss="{0:.6f}".format(valid_loss))
#
#             # Early Stopping
#             early_stopping(valid_loss, model)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#         # load the last checkpoint with the best model
#         mdl_name = 'vali.pt'
#         savemodel = os.path.join('../Checkpoint', mdl_name)
#         model.load_state_dict(torch.load(savemodel))
#         loss_dic = {
#             'train_losses': train_losses,
#             'valid_losses': valid_losses,
#         }
#         return model, loss_dic

# def train_model(model, train_loader, valid_loader, test_loader, lr, epochs,
#                 patience, scaler, win_out, rawdf, device):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     early_stopping = EarlyStopping(patience=patience, verbose=True)
#     train_losses = []
#     valid_losses = []
#
#     with (trange(epochs) as tr):
#         for epoch in tr:
#             # Training
#             model.train()
#             criterion = nn.MSELoss()
#             train_loss = 0.0
#             for data, target in train_loader:
#                 data, target = data.to(device), target.to(device)
#                 optimizer.zero_grad()
#                 mean, sigma = model(data)
#                 loss = nll_loss(mean, sigma, target.squeeze())+criterion(mean,target.squeeze())*0
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item() * data.size(0)
#
#             # Validation
#             model.eval()
#             valid_loss = 0.0
#             with torch.no_grad():
#                 for data, target in valid_loader:
#                     data, target = data.to(device), target.to(device)
#                     mean, sigma = model(data)
#                     loss = nll_loss(mean, sigma, target.squeeze())+criterion(mean,target.squeeze())*0
#                     res = 10
#                     valid_loss += loss.item() * data.size(0)
#
#             train_loss = train_loss / len(train_loader.dataset)
#             valid_loss = valid_loss / len(valid_loader.dataset)
#             train_losses.append(train_loss)
#             valid_losses.append(valid_loss)
#
#             if epoch % res == 0:
#                 device = device
#                 with torch.no_grad():
#                     for data, target in test_loader:
#                         data, target = data.to(device), target.to(device)
#                         mean, sigma = model(data)
#                         prediction_distribution = torch.distributions.Normal(mean, sigma/2)
#                         samples_dict = {}
#                         for s in range (50):
#                             samples = prediction_distribution.sample().cpu()
#                             samples = np.clip(samples, 0, None)
#                             outputs_denorm = []
#                             for idx in range(samples.shape[0]):
#                                 outputs_denorm.append(
#                                     scaler.inverse_transform(samples[[idx], :].reshape(-1, 1)))
#                             samples_dict[s] = outputs_denorm
#
#
#                 test_len = len(outputs_denorm)
#                 pred_len = win_out
#                 display = True
#                 if display == True :
#                     fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.8), dpi=300, constrained_layout=True)
#                     ax.plot_date(rawdf.index[:test_len], rawdf.iloc[:,[1]].values[:test_len],
#                                  '-', linewidth=1, color="#159A9C", label='Groundtruth')
#                     timestep = 0
#                     y_true = (rawdf.iloc[:, [1]].values[:win_out][:test_len]).reshape(-1, 1)
#                     mae = 0
#                     avg = 0
#                     for s in range (50):
#                         tem = samples_dict[s][timestep][:test_len - timestep]
#                         tem = tem
#                         y_pred = tem
#                         mae += mean_absolute_error(y_true, y_pred)
#                         avg += tem
#                         ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
#                                          tem, '--', linewidth=0.1, color="gray")
#                     ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
#                                  avg/(s+1), '-', linewidth=0.5, color="red", label='CNN-LSTM')
#                     text_gray = 'Epochs:{}\nMAE:{:.2f}'.format(epoch, mae/(s+1))
#                     ax.text(0.05, 0.8, text_gray, fontsize=6, color='gray', fontweight='bold', transform=ax.transAxes)
#                     ax.tick_params(axis='both', which='minor', labelsize=7)
#                     ax.tick_params(axis='both', which='major', labelsize=7)
#                     ax.xaxis.set_minor_locator(dates.HourLocator(byhour=range(2, 24, 4)))
#                     ax.xaxis.set_minor_formatter(dates.DateFormatter("%H"))
#                     ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
#                     ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
#                     ax.set_xlabel(None)
#                     ax.set_ylabel('Target', fontsize=7)
#                     ax.margins(x=0)
#                     ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=6, frameon=False)
#                     plt.show()
#                 else:
#                     0
#
#             tr.set_postfix(epoch="{0:.0f}".format(epoch+1), train_loss="{0:.6f}".format(train_loss),
#                            valid_loss="{0:.6f}".format(valid_loss))
#
#             # Early Stopping
#             early_stopping(valid_loss, model)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break
#
#         # load the last checkpoint with the best model
#         mdl_name = 'vali.pt'
#         savemodel = os.path.join('../Checkpoint', mdl_name)
#         model.load_state_dict(torch.load(savemodel))
#         loss_dic = {
#             'train_losses': train_losses,
#             'valid_losses': valid_losses,
#         }
#         return model, loss_dic