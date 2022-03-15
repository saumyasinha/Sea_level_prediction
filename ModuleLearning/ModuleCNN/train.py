import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from ModuleLearning.ModuleCNN.Model import trainBatchwise as trainconv,FullyConvNet
from ModuleLearning.ModuleCNN.convLSTM import trainBatchwise as trainconvlstm,ConvLSTM
from ModuleLearning import eval

def get_target_mask(y):

    # missing_val = 1e+36
    print(np.isnan(y).sum())
    mask = ~np.isnan(y)#(y != missing_val)
    print("mask",mask.shape, y[mask].max(),y[mask].min())
    # print("num of ocean pixels: ", mask.sum())
    y_copy = y.copy()
    y_copy[np.isnan(y_copy)] = 0
    return y_copy, mask


def basic_CNN_train(X_train, y_train, X_valid, y_valid, weight_map, n_features, n_timesteps, epochs, batch_size, learning_rate, folder_saving, model_saved, include_heat, quantile, alphas, model_type,  pretrained_model_path=None, hidden_dim=15, num_layers=1,kernel_size = (3,3), attention = False, n_predictions =1):

    valid = True
    outputs_quantile = len(alphas)

    y_train, train_mask = get_target_mask(y_train)
    y_valid, valid_mask = get_target_mask(y_valid)

    X_train, y_train, train_mask, weight_map = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(train_mask), torch.from_numpy(weight_map)
    X_valid, y_valid, valid_mask = torch.from_numpy(X_valid), torch.from_numpy(y_valid), torch.from_numpy(valid_mask)

    if include_heat is False:
        print(X_train.shape)
        X_train = X_train.permute(0, 3, 1, 2)
        print(X_train.shape)
        X_valid = X_valid.permute(0, 3, 1, 2)

    else:
        print(X_train.shape)
        X_train = X_train.permute(0, 3, 4, 1, 2)
        print(X_train.shape)
        X_valid = X_valid.permute(0, 3, 4, 1, 2)

    if model_type != "ConvLSTM":
        if include_heat:
            current_heat_train = X_train[:,-1,1,:,:]
            X_train_reduced = X_train[:,:,0,:,:]
            X_train = np.concatenate([X_train_reduced, current_heat_train[:,np.newaxis,:,:]], axis=1)

            current_heat_valid = X_valid[:, -1, 1, :, :]
            X_valid_reduced = X_valid[:, :, 0, :, :]
            X_valid = np.concatenate([X_valid_reduced, current_heat_valid[:, np.newaxis, :, :]], axis=1)

            print(X_train.shape, X_valid.shape)

        if model_type[-2:]=="3d":
            X_train = X_train[:,  np.newaxis, :, :, :]
            X_valid = X_valid[:, np.newaxis, :,: , :]
        train_loss, valid_loss = trainconv(model_type, X_train, y_train, X_valid, y_valid,  weight_map, train_mask, valid_mask,
                                                n_predictions, n_features, n_timesteps, epochs, batch_size, learning_rate, folder_saving, model_saved, quantile,
                                                alphas=np.arange(0.05, 1.0, 0.05), outputs_quantile=outputs_quantile, valid=valid, pretrained_model_path=pretrained_model_path, patience=1000)

    else:
        if include_heat is False:
            X_train = X_train[:,:,np.newaxis,:,:]
            X_valid = X_valid[:, :, np.newaxis, :, :]

        print(X_train.shape)

        train_loss, valid_loss = trainconvlstm(X_train, y_train, X_valid, y_valid, weight_map, train_mask, valid_mask,
                                           n_predictions, n_features, n_timesteps, epochs, batch_size, learning_rate,
                                           folder_saving, model_saved, quantile,attention,
                                           alphas=np.arange(0.05, 1.0, 0.05), outputs_quantile=outputs_quantile,
                                           valid=valid, hidden_dim=hidden_dim, num_layers=num_layers, kernel_size=kernel_size, patience=1000)

    loss_plots(train_loss, valid_loss, folder_saving, model_saved)

    # return train_mask, valid_mask


def basic_CNN_test(X_train, y_train, X_valid, y_valid, X_test, y_test, weight_map_wo_patches, n_features, n_timesteps,folder_saving, model_saved, quantile, alphas,model_type, hidden_dim=15, num_layers=1, kernel_size=(3,3),attention = False, n_predictions = 1):


    if X_valid is not None:
        X_valid = torch.from_numpy(X_valid)
        X_valid = X_valid.permute(0, 3, 1, 2)
        # y_valid, valid_mask = get_target_mask(y_valid)
        if model_type == "ConvLSTM":
            X_valid = X_valid[:, :, np.newaxis, :, :]

    if X_test is not None:
        X_test = torch.from_numpy(X_test)
        X_test = X_test.permute(0, 3, 1, 2)
        if model_type == "ConvLSTM":
            X_test = X_test[:, :, np.newaxis, :, :]

    valid_rmse, valid_mae, test_rmse, test_mae, valid_mask,test_mask = 0,0,0,0,None,None

    outputs_quantile = len(alphas)
    print(torch.cuda.is_available())

    if model_type != "ConvLSTM":
        if model_type[-2:]=="3d":
            X_valid = X_valid[:,  np.newaxis, :, :, :]
            X_test = X_test[:, np.newaxis, :,: , :]

        basic_forecaster = FullyConvNet(model_type, quantile, outputs_quantile, n_timesteps)
    else:
        basic_forecaster = ConvLSTM(input_dim=n_features,
                                    hidden_dim=hidden_dim,
                                    kernel_size=kernel_size,
                                    num_layers=num_layers,
                                    attention=attention,
                                    batch_first=True,
                                    bias=True,
                                    return_all_layers=False)

    # basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved)) #, map_location=torch.device('cpu'))
    #if torch.cuda.is_available():
     #   basic_forecaster = basic_forecaster.cuda()
      #  basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved))
    #else:
    basic_forecaster.load_state_dict(torch.load(folder_saving + model_saved, map_location=torch.device('cpu')))

    basic_forecaster.eval()


    ##saving the model's results in the altimeter period and 30 yrs in advance
    # X_train = torch.from_numpy(X_train)
    # X_train = X_train.permute(0, 3, 1, 2)
    # X_altimeter = torch.cat((X_train[-17*12:, :,:,:], X_valid[:9*12,:,:,:]),0)
    # np.save(folder_saving + "/" + "climate_model_1994_2019.npy", X_altimeter[:,-1,:,:])
    # print(X_altimeter.shape,  X_altimeter[:,-1,:,:].shape)
    # y_altimeter = np.concatenate((y_train[-17*12:, :,:], y_valid[:9*12,:,:]),axis=0)
    # print(y_altimeter.shape)
    # np.save(folder_saving + "/" + "true_climate_model_2024-2049.npy", y_altimeter)
    # y_altimeter_pred = basic_forecaster.forward(X_altimeter)
    # y_altimeter_pred = y_altimeter_pred.cpu().detach().numpy()
    # print(y_altimeter_pred.shape)
    # print(y_altimeter_pred[0,0,0])
    # np.save(folder_saving + "/" + "predictions_on_climate_model_2024-2049.npy", y_altimeter_pred)

    ##saving train predictions
    # X_train = torch.from_numpy(X_train)
    # X_train = X_train.permute(0, 3, 1, 2)
    # if model_type == "ConvLSTM":
    #     X_train = X_train[:, :, np.newaxis, :, :]
    #     last_states = basic_forecaster.forward(X_train)
    #     y_train_pred = last_states[0][0]
    #
    # else:
    #     y_train_pred = basic_forecaster.forward(X_train)
    #
    # y_train_pred = y_train_pred.cpu().detach().numpy()
    # np.save(folder_saving + "/" + "train_predictions.npy", y_train_pred)

    if X_test is not None:
    #    if torch.cuda.is_available():
     #       X_test = X_test.cuda()
        if model_type != "ConvLSTM":
            y_pred = basic_forecaster.forward(X_test)
        else:
            last_states = basic_forecaster.forward(X_test)
            y_pred = last_states[0][0]  # 0 for layer index, 0 for h index
        # testLoss = MaskedMSELoss(y_pred, y_test, test_mask)
        y_pred = y_pred.cpu().detach().numpy()
        print(y_pred.shape)
        y_test_wo_patches = y_test #eval.combine_image_patches(y_test)
        y_pred_wo_patches = y_pred #eval.combine_image_patches(y_pred)

        if y_test is not None:
            np.save(folder_saving + "/" + "test_predictions.npy", y_pred_wo_patches)
            y_test_wo_patches, test_mask = get_target_mask(y_test_wo_patches)
            test_rmse, test_mae = eval.evaluation_metrics(y_pred_wo_patches, y_test_wo_patches, test_mask, weight_map_wo_patches)

        else:
            np.save(folder_saving + "/" + "altimeter_predictions.npy", y_pred_wo_patches)
        # print("test rmse and mae scores: ", test_rmse, test_mae)


    if X_valid is not None:
      #  if torch.cuda.is_available():
       #     X_valid = X_valid.cuda()
        if model_type != "ConvLSTM":
            y_valid_pred = basic_forecaster.forward(X_valid)
        else:
            last_states = basic_forecaster.forward(X_valid)
            y_valid_pred = last_states[0][0]  # 0 for layer index, 0 for h index
        # testLoss = MaskedMSELoss(y_pred, y_test, test_mask)
        # validLoss = MaskedMSELoss(y_valid_pred, y_valid, valid_mask)
        y_valid_pred = y_valid_pred.cpu().detach().numpy()
        print(y_valid_pred.shape)

       #
        y_valid_wo_patches = y_valid #eval.combine_image_patches(y_valid)
        y_valid_pred_wo_patches = y_valid_pred #eval.combine_image_patches(y_valid_pred)
        np.save(folder_saving + "/" + "valid_predictions.npy", y_valid_pred_wo_patches)
        y_valid_wo_patches, valid_mask = get_target_mask(y_valid_wo_patches)
        valid_rmse, valid_mae = eval.evaluation_metrics(y_valid_pred_wo_patches, y_valid_wo_patches, valid_mask, weight_map_wo_patches)

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









