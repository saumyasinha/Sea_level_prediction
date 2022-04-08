import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
from ModuleLearning import preprocessing, eval
from ModuleLearning.ModuleCNN import train as train_cnn

path_local = "/Users/saumya/Desktop/Sealevelrise/"
path_cluster = "/pl/active/machinelearning/Saumya/ML_for_sea_level/"
path_project = path_cluster

path_data = path_project+"Data/"
path_models = path_project+"ML_Models/"
path_data_fr = path_data + "Forced_Responses/"

## which climate model to work on
models = ['CESM1LE','CESM2LE'] #['CESM1LE'] # ['MIROC-ES2L'] #['MPI-ESM1-2-HR']

path_sealevel_folder = path_data_fr + "zos/"
path_heatcontent_folder = path_data_fr + "heatfull/"

path_folder = path_sealevel_folder

historical_path = path_folder + "1850-2014/npy_files/"
future_path = path_folder + "2015-2100/npy_files/"

historical_heat_path = path_heatcontent_folder + "1850-2014/npy_files/"
future_heat_path = path_heatcontent_folder + "2015-2100/npy_files/"

## parameters
train_start_year = 1930 # 1880 #
train_end_year = 2040 #1990 #
test_start_year = 2041 #1991
test_end_year = 2070 #2020 #

## For the case of trends, we end up training from 1930-2030 and testing on 2031-2040, the number of data points
## is reduced due to taking a trend over 30 years, so we only end up having trend data till 2070 that becomes the label of the
## last test point.

lead_years = 30
model_type = "Unet" #"DilatedUnet3d"#"Unet"#"SmaAT_Unet" #"DilatedUnet"#"Unet_Attn" #"ConvLSTM" #

## if we want to have probabilsitic prediction
quantile = False
alphas = np.arange(0.05, 1.0, 0.05)
q50 = 9

## folders to finally save the model
reg = "CNN/Unet/"# Unet"


# sub_reg = "_rerun_cnn_with_1yr_lag_large_batchnorm_unet_downscaled_weighted_changed_years_not_normalized"#cnn_with_1yr_lag_large_batchnorm_unet_attn_downscaled_weighted_changed_years_not_normalized"#"
# sub_reg = "cnn_with_1yr_lag_large_batchnorm_unet_attn_downscaled_weighted_changed_years_not_normalized"
sub_reg = "_combined_cesm1and2_trend_cnn_with_1yrlag_large_batchnorm_unet_downscaled_weighted_changed_years_not_normalized"

## Hyperparameters
hidden_dim = 12 #40 #24
num_layers=1 #1

kernel_size = (3,3)

batch_size = 6
epochs = 200#200
lr = 1e-4

features = ["sea_level"]
n_features = len(features)
n_prev_months = 12 ##seq-length of the deep sequence models

downscaling = True #converting 360*180 to 180*90
include_heat = False
attention = False


def main():

    X_train_combined = []
    X_valid_combined = []
    X_test_combined = []
    y_train_combined = []
    y_valid_combined = []
    y_test_combined = []

    ## these weights are cosine of latitude, important for using weighted RMSE as a loss function
    weight_map = np.load(historical_path + "weights_historical_CESM1LE_zos_fr_1850_2014.npy")
    weight_map = np.abs(weight_map)
    if downscaling:
        weight_map = block_reduce(weight_map, (2, 2), np.mean)  # (2,2)
    print(weight_map.shape, np.max(weight_map), np.min(weight_map))

    folder_saving = path_models + "/combined/" + reg + "/" + sub_reg + "/"
    os.makedirs(
        folder_saving, exist_ok=True)

    f = open(folder_saving + "/results.txt", 'a')

    for model in models:

        train, test = preprocessing.create_train_test_split(model, historical_path, future_path, train_start_year, train_end_year, test_start_year, test_end_year, n_prev_months, lead_years, downscaling)
        X_train, y_train, X_test, y_test = preprocessing.create_train_test_and_labels_on_trends(train, test, n_prev_months, path_models + "/combined/")
        ## the values are in cms
    #
        ## splitting train into train+valid
        split_index = 20*12

        n_prev_times = n_prev_months

        ## remove land values
        X_train = preprocessing.remove_land_values(X_train)
        X_test = preprocessing.remove_land_values(X_test)

        ## add previous timestep values
        X_train = preprocessing.include_prev_timesteps(X_train, n_prev_times)
        X_test = preprocessing.include_prev_timesteps(X_test, n_prev_times)

        y_train = y_train[:,:,n_prev_times:]
        y_test = y_test[:,:,n_prev_times:]

        y_train = np.transpose(y_train, (2,0,1)) #y_train.reshape(-1,lon,lat)
        y_test = np.transpose(y_test, (2,0,1)) #y_test.reshape(-1, lon, lat)


        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print(np.max(X_train),np.max(y_train))

        train_valid_split_index = len(X_train) - split_index #keeping later 30 years for validation
        X,y = X_train,y_train
        X_train = X[:train_valid_split_index]
        y_train = y[:train_valid_split_index]
        X_valid = X[train_valid_split_index:]
        y_valid = y[train_valid_split_index:]


        print("train/valid sizes: ", len(X_train), " ", len(X_valid))

        # X_train, X_valid, X_test = preprocessing.normalize_from_train(X_train,X_valid, X_test)

        X_train_combined.append(X_train)
        X_valid_combined.append(X_valid)
        X_test_combined.append(X_test)
        y_train_combined.append(y_train)
        y_valid_combined.append(y_valid)
        y_test_combined.append(y_test)


    X_train_combined_input = np.concatenate(X_train_combined)
    X_valid_combined_input = np.concatenate(X_valid_combined)
    X_test_combined_input = np.concatenate(X_test_combined)
    y_train_combined_input = np.concatenate(y_train_combined)
    y_valid_combined_input = np.concatenate(y_valid_combined)
    y_test_combined_input = np.concatenate(y_test_combined)

    print("combined train input shapes: ", X_train_combined_input.shape, y_train_combined_input.shape)
    print("combined valid input shapes: ", X_valid_combined_input.shape, y_valid_combined_input.shape)
    print("combined test input shapes: ", X_test_combined_input.shape, y_test_combined_input.shape)

    model_saved = "model_at_lead_"+str(lead_years)+"_yrs"
    train_cnn.basic_CNN_train(X_train_combined_input, y_train_combined_input, X_valid_combined_input, y_valid_combined_input, weight_map, n_features,  n_prev_times+1, epochs, batch_size, lr, folder_saving, model_saved, include_heat, quantile, alphas, model_type = model_type, hidden_dim = hidden_dim, num_layers = num_layers, kernel_size=kernel_size, attention = attention)
    valid_rmse, valid_mae, test_rmse, test_mae, valid_mask, test_mask = train_cnn.basic_CNN_test(X_train_combined_input, y_train_combined_input, X_valid_combined_input, y_valid_combined_input, X_test_combined_input, y_test_combined_input, weight_map, n_features, n_prev_times+1, folder_saving, model_saved, quantile, alphas, model_type = model_type, hidden_dim = hidden_dim, num_layers = num_layers, kernel_size=kernel_size, attention=attention)
    f.write('\n evaluation metrics (rmse, mae) on valid data ' + str(valid_rmse) + "," + str(valid_mae) +'\n')
    f.write('\n evaluation metrics (rmse, mae) on test data ' + str(test_rmse) + "," + str(test_mae) + '\n')
    f.close()




if __name__=='__main__':
    main()

