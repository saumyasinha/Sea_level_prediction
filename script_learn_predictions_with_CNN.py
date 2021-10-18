import os
import numpy as np
from sklearn.model_selection import train_test_split
# from Sea_level_prediction.ModuleLearning import preprocessing,eval
# from Sea_level_prediction.ModuleLearning.ModuleCNN import train as train_cnn
from ModuleLearning import preprocessing, eval
from ModuleLearning.ModuleCNN import train as train_cnn

path_local = "/Users/saumya/Desktop/Sealevelrise/"
path_cluster = "/pl/active/machinelearning/ML_for_sea_level/"
path_project = path_cluster
path_data = path_project+"Data/"
path_models = path_project+"ML_Models/"
path_data_fr = path_data + "Forced_Responses/"

models = ['MPI-ESM1-2-HR'] #,'MPI-ESM1-2-LR', 'ACCESS-ESM1-5','MIROC-ES2L']

path_sealevel_folder = path_data_fr + "zos/"
path_heatcontent_folder = path_data_fr + "heatfull/"

path_folder = path_sealevel_folder

historical_path = path_folder + "1850-2014/"
future_path = path_folder + "2015-2100/"

train_start_year = 1900
train_end_year = 1990
test_start_year = 1991
test_end_year = 2020

lead_years = 30
quantile = False
alphas = np.arange(0.05, 1.0, 0.05)
q50 = 9
reg = "CNN"

sub_reg = "cnn_with_1yr_lag_small_channels_with_dropout"
## Hyperparameters
features = ["sea_level"]
n_features = len(features)
n_prev_months = 12


batch_size = 16
epochs = 250

lr = 1e-4



def main():

    for model in models:

        folder_saving = path_models+ model + "/" + reg + "/"+ sub_reg + "/"
        os.makedirs(
            folder_saving, exist_ok=True)

        f = open(folder_saving +"/results.txt", 'a')

        train, test = preprocessing.create_train_test_split(model, historical_path, future_path, train_start_year, train_end_year, test_start_year, test_end_year, n_prev_months, lead_years)
        # np.save(path_data_fr+ model + "/"+"train_for_"+str(train_start_year)+"-"+str(train_end_year)+".npy", train)
        # np.save(path_data_fr+ model + "/"+"test_for_" + str(test_start_year) + "-" + str(test_end_year) + ".npy", test)

        X_train, y_train, X_test, y_test = preprocessing.create_labels(train, test, lead_years)

        ## remove land values
        X_train = preprocessing.remove_land_values(X_train)
        X_test = preprocessing.remove_land_values(X_test)

        ## add previous timestep values
        X_train = preprocessing.include_prev_timesteps(X_train, n_prev_months)
        X_test = preprocessing.include_prev_timesteps(X_test, n_prev_months)

        y_train = y_train[:,:,n_prev_months:]
        y_test = y_test[:,:,n_prev_months:]

        lon = y_train.shape[0]
        lat = y_train.shape[1]
        y_train = y_train.reshape(-1,lon,lat)
        y_test = y_test.reshape(-1, lon, lat)

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        # X_train, X_valid, y_train, y_valid = train_test_split(
        #     X_train, y_train, test_size=0.2, random_state=42)

        train_valid_split_index = len(X_train) - 120 #keeping later 10 years for validation
        X,y = X_train,y_train
        X_train = X[:train_valid_split_index]
        y_train = y[:train_valid_split_index]
        X_valid = X[train_valid_split_index:]
        y_valid = y[train_valid_split_index:]

        print("train/valid sizes: ", len(X_train), " ", len(X_valid))

        X_train_w_patches,y_train_w_patches = preprocessing.get_image_patches(X_train,y_train)
        X_valid_w_patches, y_valid_w_patches = preprocessing.get_image_patches(X_valid, y_valid)
        X_test_w_patches, y_test_w_patches = preprocessing.get_image_patches(X_test, y_test)

        model_saved = "model_at_lead_"+str(lead_years)+"_yrs"

        train_cnn.basic_CNN_train(X_train_w_patches, y_train_w_patches, X_valid_w_patches, y_valid_w_patches, n_features, n_prev_months+1, epochs, batch_size, lr, folder_saving, model_saved, quantile, alphas)
        y_valid_w_patches_copy = y_valid_w_patches.copy() #if you are not doing this then pass X_valid and y_valid as None
        valid_rmse, valid_mae, test_rmse, test_mae, test_mask = train_cnn.basic_CNN_test(X_valid_w_patches, y_valid_w_patches_copy, X_test_w_patches, y_test_w_patches, n_features, n_prev_months+1, folder_saving, model_saved, quantile, alphas)

        f.write('\n evaluation metrics (rmse, mae) on valid data ' + str(valid_rmse) + "," + str(valid_mae) +'\n')
        f.write('\n evaluation metrics (rmse, mae) on test data ' + str(test_rmse) + "," + str(test_mae) + '\n')
        f.close()
        # # yr_1993 = y_train[63*12]
        # y_pred = np.load(folder_saving+"/test_predictions.npy")
        # print(y_pred.shape)
        # yr_JAN2021 = y_test[0]
        # yr_JAN2021_pred =y_pred[0]
        # eval.plot(yr_JAN2021, test_mask[0], "model_2021JAN_sla")
        # eval.plot(yr_JAN2021_pred, test_mask[0], "predicted_2021JAN_sla")


if __name__=='__main__':
    main()
