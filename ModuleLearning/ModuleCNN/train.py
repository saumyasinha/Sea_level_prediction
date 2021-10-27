import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
# from Sea_level_prediction.ModuleLearning.ModuleCNN.Model import trainBatchwise,FullyConvNet,MaskedMSELoss,MaskedL1Loss
# from Sea_level_prediction.ModuleLearning import eval
from ModuleLearning.ModuleCNN.Model import trainBatchwise,FullyConvNet,MaskedMSELoss,MaskedL1Loss
from ModuleLearning import eval

def get_target_mask(y):
    # print(y[:5,0,0])
    missing_val = 1e+36
    mask = (y != missing_val)
    print("mask",mask.shape, y[mask].max(),y[mask].min())
    # print("num of ocean pixels: ", mask.sum())
    # y[y == missing_val] = 0
    return y, mask

def basic_CNN_train(X_train, y_train, X_valid, y_valid, n_features, n_timesteps, epochs, batch_size, learning_rate, folder_saving, model_saved, quantile, alphas, n_predictions =1):
    valid = True
    outputs_quantile = len(alphas)

    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_valid, y_valid = torch.from_numpy(X_valid), torch.from_numpy(y_valid)

    X_train = X_train.permute(0,3, 1, 2)
    X_valid = X_valid.permute(0,3, 1, 2)

    y_train, train_mask = get_target_mask(y_train)
    y_valid, valid_mask = get_target_mask(y_valid)

    train_loss, valid_loss = trainBatchwise(X_train, y_train, X_valid, y_valid, train_mask, valid_mask,
                                            n_predictions, n_features, n_timesteps, epochs, batch_size, learning_rate, folder_saving, model_saved, quantile,
                                            alphas=np.arange(0.05, 1.0, 0.05), outputs_quantile=outputs_quantile, valid=valid, patience=1000)
    loss_plots(train_loss, valid_loss, folder_saving, model_saved)

    # return train_mask, valid_mask


def basic_CNN_test(X_valid, y_valid, X_test, y_test, n_features, n_timesteps,folder_saving, model_saved, quantile, alphas, n_predictions = 1):

    if X_valid is not None:
        X_valid = torch.from_numpy(X_valid)
        X_valid = X_valid.permute(0, 3, 1, 2)
        # y_valid, valid_mask = get_target_mask(y_valid)

    if X_test is not None:
        X_test = torch.from_numpy(X_test)
        X_test = X_test.permute(0, 3, 1, 2)

    valid_rmse, valid_mae, test_rmse, test_mae, test_mask = 0,0,0,0,None

    outputs_quantile = len(alphas)
    print(torch.cuda.is_available())
    basic_forecaster = FullyConvNet(quantile, outputs_quantile, n_timesteps)

    # basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved)) #, map_location=torch.device('cpu'))
    #if torch.cuda.is_available():
      #  basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved))
    #if not torch.cuda.is_available():
    basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved, map_location=torch.device('cpu')))

    basic_forecaster.eval()
    
    if X_test is not None:
    #    if torch.cuda.is_available():
     #       X_test = X_test.cuda()
        y_pred = basic_forecaster.forward(X_test)
        # testLoss = MaskedMSELoss(y_pred, y_test, test_mask)
        y_pred = y_pred.cpu().detach().numpy()
        print(y_pred.shape)

        y_test_wo_patches = eval.combine_image_patches(y_test)
        y_pred_wo_patches = eval.combine_image_patches(y_pred)
        np.save(folder_saving + "/" + "test_predictions.npy", y_pred_wo_patches)
        y_test_wo_patches, test_mask = get_target_mask(y_test_wo_patches)
        test_rmse, test_mae = eval.evaluation_metrics(y_pred_wo_patches, y_test_wo_patches, test_mask)

        # print("test rmse and mae scores: ", test_rmse, test_mae)


    if X_valid is not None:
      #  if torch.cuda.is_available():
       #     X_valid = X_valid.cuda()
        y_valid_pred = basic_forecaster.forward(X_valid)
        # validLoss = MaskedMSELoss(y_valid_pred, y_valid, valid_mask)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()
        print(y_valid_pred.shape)

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









