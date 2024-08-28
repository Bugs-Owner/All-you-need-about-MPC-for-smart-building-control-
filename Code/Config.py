def paras(args):
    para = {}
    para["lr"], para["epochs"] = args["lr"], args["epochs"]
    para["training_batch"]=args['training_batch']
    para['win_out'] = args["win_out"]
    para['win_in'] = args["win_in"]
    para["patience"] = args["patience"]
    # para["CNN_in"] = 7
    # para["CNN_out"] = 12
    # para["CNN_kernel_size"] = 4
    # para["CNN_stride"] = 1
    # para["CNN_padding"] = 1
    # para["Pool_kernel_size"] = 2
    # para["Pool_stride"] = 1
    # para["LSTM_in"] = para["CNN_out"]
    # para["LSTM_hidd"] = 64
    # para["num_layers"] = 1
    # para["FC_in"] = para["LSTM_hidd"]
    # para["FC_out"] = args["win_out"]
    # para["Encoder_hidd"] = 32
    # para["Decoder_hidd"] = 32
    # para["BLSTM_input"] = para["CNN_out"]
    # para["BLSTM_output"] = args["win_out"]

    para["input_dim"] = 7
    para["cnn_channels"] = 16
    para["cnn_kernel_size"] = 2
    para["lstm_num_layers"] = 1
    para["decoder_input_dim"] = 7
    para["lstm_encoder"] = 64
    para["lstm_decoder"] = 64
    return para
