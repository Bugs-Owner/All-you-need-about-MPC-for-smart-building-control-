import time
from DataMaker import DataCook
from Config import paras as mdl_para
import torch
import pandas as pd
from Model import mdl, BLSTM, BCLSTM
from Play import train_model
import numpy as np

class TS:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.para = None
        self.args = None

    def data_ready(self, args):
        print('Preparing data')
        start_time = time.time()
        #Step 1, load csv file
        df=pd.read_csv(args['datapath'])
        df.index = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M")
        df["Time"] = df.index
        h = df["Time"].dt.hour
        m = df["Time"].dt.minute
        mon = df["Time"].dt.month
        day = df["Time"].dt.day
        ti = h + m / 60
        df["day_sin"] = np.sin(ti * (2. * np.pi / 24))
        df["day_cos"] = np.cos(ti * (2. * np.pi / 24))
        df["mon_sin"] = np.sin(mon * (2. * np.pi / 12))
        df["mon_cos"] = np.cos(mon * (2. * np.pi / 12))
        df["date_sin"] = np.sin(day * (2. * np.pi / 31))
        df["date_cos"] = np.cos(day * (2. * np.pi / 31))

        df_cleaned = df.copy()
        df_cleaned.index = pd.to_datetime(df_cleaned.index)
        df_cleaned = df_cleaned.groupby(df.index.date).filter(lambda x: len(x) == 24)

        #df.resample()
        DC = DataCook(df=df_cleaned)
        DC.data_preprocess()
        DC.data_roll(args=args)
        DC.data_loader(batch=args["training_batch"])
        para = mdl_para(args=args)
        self.dataset = DC
        self.args = args
        self.device = torch.device(self.args["device"])
        para['device'] = self.device
        self.para = para
        print("--- %s seconds ---" % (time.time() - start_time))

    def train(self):
        device = torch.device(self.args["device"] if torch.cuda.is_available() else "cpu")
        #model = mdl.CNN_LSTM(par = self.para).to(device)
        #model = BLSTM.BayesianLSTM(par=self.para,device=device).to(device)
        model = BCLSTM.EncoderDecoderCNNLSTM(par=self.para).to(device)

        start_time = time.time()
        print('Training')

        self.model, self.loss_dic = train_model(model=model,
                                                train_loader=self.dataset.TrainLoader,
                                                valid_loader=self.dataset.ValidLoader,
                                                test_loader=self.dataset.TestLoader,
                                                lr=self.para['lr'],
                                                epochs=self.para['epochs'],
                                                patience=self.para['patience'],
                                                scaler=self.dataset.Scaler,
                                                win_out=self.para['win_out'],
                                                rawdf=self.dataset.test_raw_df,
                                                device=self.device)
        print("--- %s seconds ---" % (time.time() - start_time))
