from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import torch

class MyData(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = np.array(X_seq)
        self.y = np.array(y_seq)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X[index]
        targets = self.y[index][:,[0]]
        return torch.from_numpy(features).float(), torch.from_numpy(targets).float()



class DataCook:
    def __init__(self, df):
        self.df = df

    def data_preprocess(self):
        Scaler = MinMaxScaler(feature_range=(0, 1))
        data = self.df.iloc[:,[1]].values.reshape(-1, 1)
        day_sin = self.df["day_sin"].values.reshape(-1, 1)
        day_cos = self.df["day_cos"].values.reshape(-1, 1)
        date_sin = self.df["date_sin"].values.reshape(-1, 1)
        date_cos = self.df["date_cos"].values.reshape(-1, 1)
        mon_sin = self.df["mon_sin"].values.reshape(-1, 1)
        mon_cos = self.df["mon_cos"].values.reshape(-1, 1)
        Scaled = Scaler.fit_transform(data).reshape(-1, 1)
        self.Scaler = Scaler
        self.Processed_data = np.concatenate((Scaled, day_sin, day_cos, date_sin, date_cos, mon_sin, mon_cos), axis=1)

    def data_roll(self, args):
        win_in = args["win_in"]
        win_out = args["win_out"]
        startday = args["startday"]
        trainday = args["trainday"]
        testday = args["testday"]
        resolution = args["resolution"]
        self.resolution = resolution
        self.startday = startday
        self.trainday = trainday
        self.testday = testday
        self.win_in = win_in
        self.win_out = win_out
        res = int(1440 / self.resolution)
        self.X_train = self.Processed_data[res * (self.startday) : res * (self.startday + self.trainday)]
        self.X_test = self.Processed_data[res * (self.startday + self.trainday) - self.win_in : res * (self.startday + self.trainday + self.testday) + self.win_out]
        self.test_raw_df = self.df[res * (self.startday + self.trainday):res * (self.startday + self.trainday + self.testday + 1)]
        self.test_start = self.test_raw_df.index[0].strftime("%m-%d")
        self.test_end = self.test_raw_df.index[-1].strftime("%m-%d")

    def data_loader(self, batch):
        self.batch = batch
        # Training loader
        L = len(self.X_train)
        X, y = [], []
        for i in range(L - self.win_in - self.win_out):
            train_seq = self.X_train[i: i + self.win_in + self.win_out]
            train_label = self.X_train[i + self.win_in: i + self.win_in + self.win_out]
            X.append(train_seq)
            y.append(train_label)
        myset = MyData(X, y)
        train_size = int(0.7 * len(myset))
        valid_size = len(myset) - train_size
        train_dataset, valid_dataset = random_split(myset, [train_size, valid_size])
        train_params = {'batch_size': self.batch,
                        'shuffle': True}
        vali_params = {'batch_size': self.batch,
                       'shuffle': True}
        self.TrainLoader = DataLoader(train_dataset, **train_params)
        self.ValidLoader = DataLoader(valid_dataset, **vali_params)

        # Testing loader
        L = len(self.X_test)
        X, y = [], []
        for i in range(L - self.win_in - self.win_out):
            test_seq = self.X_test[i: i + self.win_in + self.win_out]
            test_label = self.X_test[i + self.win_in: i + self.win_in + self.win_out]
            X.append(test_seq)
            y.append(test_label)
        myset = MyData(X, y)
        params = {'batch_size': L - self.win_in - self.win_out,
                  'shuffle': False}
        self.TestLoader = DataLoader(myset, **params)
