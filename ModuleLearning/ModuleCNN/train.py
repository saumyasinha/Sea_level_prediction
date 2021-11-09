import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
# from Sea_level_prediction.ModuleLearning.ModuleCNN.Model import trainBatchwise as trainconv,FullyConvNet
# from Sea_level_prediction.ModuleLearning.ModuleCNN.convLSTM import trainBatchwise as trainconvlstm,ConvLSTM
# from Sea_level_prediction.ModuleLearning import eval
from ModuleLearning.ModuleCNN.Model import trainBatchwise as trainconv,FullyConvNet
from ModuleLearning.ModuleCNN.convLSTM import trainBatchwise as trainconvlstm,ConvLSTM
from ModuleLearning import eval

def get_target_mask(y):
    # print(y[:5,0,0])
    missing_val = 1e+36
    mask = ~np.isnan(y)#(y != missing_val)
    print("mask",mask.shape, y[mask].max(),y[mask].min())
    # print("num of ocean pixels: ", mask.sum())
    y[np.isnan(y)] = 0
    return y, mask

def basic_CNN_train(X_train, y_train, X_valid, y_valid, n_features, n_timesteps, epochs, batch_size, learning_rate, folder_saving, model_saved, quantile, alphas, convlstm = False, hidden_dim=15, num_layers=1,n_predictions =1):
    valid = True
    outputs_quantile = len(alphas)

    y_train, train_mask = get_target_mask(y_train)
    y_valid, valid_mask = get_target_mask(y_valid)

    X_train, y_train, train_mask = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(train_mask)
    X_valid, y_valid, valid_mask = torch.from_numpy(X_valid), torch.from_numpy(y_valid), torch.from_numpy(valid_mask)

    X_train = X_train.permute(0, 3, 1, 2)
    X_valid = X_valid.permute(0, 3, 1, 2)

    if convlstm==False:

        train_loss, valid_loss = trainconv(X_train, y_train, X_valid, y_valid, train_mask, valid_mask,
                                                n_predictions, n_features, n_timesteps, epochs, batch_size, learning_rate, folder_saving, model_saved, quantile,
                                                alphas=np.arange(0.05, 1.0, 0.05), outputs_quantile=outputs_quantile, valid=valid, patience=1000)

    else:
        X_train = X_train[:,:,np.newaxis,:,:]
        X_valid = X_valid[:, :, np.newaxis, :, :]

        print(X_train.shape)

        train_loss, valid_loss = trainconvlstm(X_train, y_train, X_valid, y_valid, train_mask, valid_mask,
                                           n_predictions, n_features, n_timesteps, epochs, batch_size, learning_rate,
                                           folder_saving, model_saved, quantile,
                                           alphas=np.arange(0.05, 1.0, 0.05), outputs_quantile=outputs_quantile,
                                           valid=valid, hidden_dim=hidden_dim, num_layers=num_layers, patience=1000)

    loss_plots(train_loss, valid_loss, folder_saving, model_saved)

    # return train_mask, valid_mask


def basic_CNN_test(X_valid, y_valid, X_test, y_test, n_features, n_timesteps,folder_saving, model_saved, quantile, alphas,convlstm = False, hidden_dim=15, num_layers=1, n_predictions = 1):

    if X_valid is not None:
        X_valid = torch.from_numpy(X_valid)
        X_valid = X_valid.permute(0, 3, 1, 2)
        # y_valid, valid_mask = get_target_mask(y_valid)
        if convlstm == True:
            X_valid = X_valid[:, :, np.newaxis, :, :]

    if X_test is not None:
        X_test = torch.from_numpy(X_test)
        X_test = X_test.permute(0, 3, 1, 2)
        if convlstm == True:
            X_test = X_test[:, :, np.newaxis, :, :]

    valid_rmse, valid_mae, test_rmse, test_mae, valid_mask,test_mask = 0,0,0,0,None,None

    outputs_quantile = len(alphas)
    print(torch.cuda.is_available())

    if convlstm == False:
        basic_forecaster = FullyConvNet(quantile, outputs_quantile, n_timesteps)
    else:
        basic_forecaster = ConvLSTM(input_dim=n_features,
                                    hidden_dim=hidden_dim,
                                    kernel_size=(3, 3),
                                    num_layers=num_layers,
                                    batch_first=True,
                                    bias=True,
                                    return_all_layers=False)

    # basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved)) #, map_location=torch.device('cpu'))
    #if torch.cuda.is_available():
      #  basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved))
    #if not torch.cuda.is_available():
    basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved, map_location=torch.device('cpu')))

    basic_forecaster.eval()
    
    if X_test is not None:
    #    if torch.cuda.is_available():
     #       X_test = X_test.cuda()
        if convlstm == False:
            y_pred = basic_forecaster.forward(X_test)
        else:
            _, last_states = basic_forecaster.forward(X_test)
            y_pred = last_states[0][0]  # 0 for layer index, 0 for h index
        # testLoss = MaskedMSELoss(y_pred, y_test, test_mask)
        y_pred = y_pred.cpu().detach().numpy()
        print(y_pred.shape)
     #
        y_test_wo_patches = eval.combine_image_patches(y_test)
        y_pred_wo_patches = eval.combine_image_patches(y_pred)
        np.save(folder_saving + "/" + "test_predictions.npy", y_pred_wo_patches)
        y_test_wo_patches, test_mask = get_target_mask(y_test_wo_patches)
        test_rmse, test_mae = eval.evaluation_metrics(y_pred_wo_patches, y_test_wo_patches, test_mask)

        # print("test rmse and mae scores: ", test_rmse, test_mae)


    if X_valid is not None:
      #  if torch.cuda.is_available():
       #     X_valid = X_valid.cuda()
        if convlstm == False:
            y_valid_pred = basic_forecaster.forward(X_valid)
        else:
            _, last_states = basic_forecaster.forward(X_valid)
            y_valid_pred = last_states[0][0]  # 0 for layer index, 0 for h index
        # testLoss = MaskedMSELoss(y_pred, y_test, test_mask)
        # validLoss = MaskedMSELoss(y_valid_pred, y_valid, valid_mask)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()
        print(y_valid_pred.shape)
       #
        y_valid_wo_patches = eval.combine_image_patches(y_valid)
        y_valid_pred_wo_patches = eval.combine_image_patches(y_valid_pred)
        np.save(folder_saving + "/" + "valid_predictions.npy", y_valid_pred_wo_patches)
        y_valid_wo_patches, valid_mask = get_target_mask(y_valid_wo_patches)
        valid_rmse, valid_mae = eval.evaluation_metrics(y_valid_pred_wo_patches, y_valid_wo_patches, valid_mask)

        print("valid rmse and mae scores: ", valid_rmse, valid_mae)

    return valid_rmse, valid_mae, test_rmse, test_mae, valid_mask, test_mask


def loss_plots(train_loss, valid_loss, folder_saving, loss_type=""):
    epochs = range(1, len(train_loss)+1)
    # train_loss = train_loss[1:]
    # valid_loss = valid_loss[1:]
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training', 'validation'], loc='lower right')

    plt.savefig(folder_saving+"loss_plots_"+loss_type)
    plt.close()









