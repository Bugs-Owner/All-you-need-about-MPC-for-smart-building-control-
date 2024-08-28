from TimePred import TS

def forecast(device, lr, epochs, patience, training_batch, win_in, win_out, trainday, startday, testday, datapath):
    args={}
    args['device'] = device
    args['lr'] = lr
    args['epochs'] = epochs
    args['patience'] = patience
    args['training_batch'] = training_batch
    args['win_in'] = win_in
    args['win_out'] = win_out
    args['trainday'] = trainday
    args['startday'] = startday
    args['testday'] = testday
    args['datapath'] = datapath
    args['resolution'] = 60

    TS_Pred = TS()
    TS_Pred.data_ready(args=args)
    TS_Pred.train()

forecast(device='cuda:0',
         lr=0.01,
         epochs=500,
         patience=300,
         training_batch=2048,
         win_in=24*3,
         win_out=24,
         trainday=60,
         startday=8,
         testday=1,
         datapath='../Data/Temp.csv')