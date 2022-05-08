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
models = ['CESM1LE']#,'CESM2LE']

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
model_type = "Unet"#"Unet" #"DilatedUnet3d"#"Unet"#"SmaAT_Unet" #"DilatedUnet"#"Unet_Attn" #"ConvLSTM" #
average_over_years = 10
## if we want to have probabilsitic prediction
quantile = False
alphas = np.arange(0.05, 1.0, 0.05)
q50 = 9

## folders to finally save the model
reg = "CNN/trend_Unet/"# Unet"


#sub_reg = "_averaged_10yrs_cnn_with_0lag_batchnorm_dilatedunet_weight_decay1e-6_downscaled_weighted_changed_years_not_normalized"
sub_reg = "_trend_cnn_with_0lag_batchnorm_bigunet_weight_decay1e-6_downscaled_weighted_changed_years_not_normalized"



## Hyperparameters
hidden_dim = 12 #40 #24
num_layers=2 #1

kernel_size = (3,3)

batch_size = 6
epochs = 200#200
lr = 1e-4

features = ["sea_level"]
n_features = len(features)

n_prev_months = 0 #12 ##seq-length of the deep sequence models
max_prev_steps=12
downscaling = True #converting 360*180 to 180*90
include_heat = False
attention = False


def main():
    
    valid_prediction_list_over_multiple_runs = []
    # test_prediction_list_over_multiple_runs = []
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

    if len(models)>1:
        folder_saving = path_models + "combined_CESM1and2/" + reg + "/" + sub_reg + "/"
    else:
        folder_saving = path_models + models[0] + "/" + reg + "/" + sub_reg + "/"

    os.makedirs(
        folder_saving, exist_ok=True)

    f = open(folder_saving + "/results.txt", 'a')

    for model in models:

        train, test = preprocessing.create_train_test_split(model, historical_path, future_path, train_start_year, train_end_year, test_start_year, test_end_year, lead_years, downscaling)
        # X_train, y_train, X_test, y_test = preprocessing.create_train_test_and_labels_on_ssh_averages(train, test, path_models + model+"/",average_over_years=average_over_years)
        X_train, y_train, X_test, y_test = preprocessing.create_train_test_and_labels_on_trends(train, test,
                                                                                                      path_models + model + "/")

        print("simple checks")
        print(np.isnan(y_train).sum(), np.isnan(X_train).sum())
        print(np.isnan(y_test).sum(), np.isnan(X_test).sum())
        print(np.nansum(X_train), np.nansum(y_train), np.nansum(X_test), np.nansum(y_test))
        ## the values are in cms for trends and meters for averages
    #
        # splitting train into train+valid
        split_index = 20*12

        n_prev_times = n_prev_months

        ## remove land values
        X_train = preprocessing.remove_land_values(X_train)
        X_test = preprocessing.remove_land_values(X_test)

        print(np.isnan(y_train).sum())
        print(np.isnan(y_test).sum())
        print(np.nansum(X_train), np.nansum(y_train), np.nansum(X_test), np.nansum(y_test))

    #     ## add previous timestep values
        X_train = preprocessing.include_prev_timesteps(X_train, n_prev_times)
        X_test = preprocessing.include_prev_timesteps(X_test, n_prev_times)

        y_train = y_train[:,:,max_prev_steps:] #n_prev_times, it should always be 12 , because you always want to start you label from 1930
        y_test = y_test[:,:,max_prev_steps:]

        y_train = np.transpose(y_train, (2,0,1)) #y_train.reshape(-1,lon,lat)
        y_test = np.transpose(y_test, (2,0,1)) #y_test.reshape(-1, lon, lat)


        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


        train_valid_split_index = len(X_train) - split_index #keeping later 30 years for validation
        X,y = X_train,y_train
        X_train = X[:train_valid_split_index]
        y_train = y[:train_valid_split_index]
        X_valid = X[train_valid_split_index:]
        y_valid = y[train_valid_split_index:]

        print(np.max(X_train), np.max(y_valid))

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

    for iter in range(3):
        os.makedirs(
            folder_saving+"/"+str(iter), exist_ok=True)
        train_cnn.basic_CNN_train(X_train_combined_input, y_train_combined_input, X_valid_combined_input, y_valid_combined_input, weight_map, n_features,  n_prev_times+1, epochs, batch_size, lr, folder_saving+"/"+str(iter)+"/", model_saved, include_heat, quantile, alphas, model_type = model_type, hidden_dim = hidden_dim, num_layers = num_layers, kernel_size=kernel_size, attention = attention)
        valid_rmse, valid_mae, test_rmse, test_mae, valid_mask, test_mask = train_cnn.basic_CNN_test(X_train_combined_input, y_train_combined_input, X_valid_combined_input, y_valid_combined_input, X_test_combined_input, y_test_combined_input, weight_map, n_features, n_prev_times+1, folder_saving+"/"+str(iter)+"/", model_saved, quantile, alphas, model_type = model_type, hidden_dim = hidden_dim, num_layers = num_layers, kernel_size=kernel_size, attention=attention)
        f.write('For run: '+str(iter)+'\n')
        f.write('\n evaluation metrics (rmse, mae) on valid data ' + str(valid_rmse) + "," + str(valid_mae) +'\n')
        f.write('\n evaluation metrics (rmse, mae) on test data ' + str(test_rmse) + "," + str(test_mae) + '\n')

        # y_valid_combined_pred_per_run = np.load(folder_saving + "/"+str(iter)+ "/valid_predictions.npy")
        # print(y_valid_combined_pred_per_run.shape)
        # #
        # valid_prediction_list_over_multiple_runs.append(y_valid_combined_pred_per_run)

    f.close()
    #
    # #
    # #
    # y_valid_combined_pred = np.mean(np.stack(valid_prediction_list_over_multiple_runs,axis=0),axis=0)
    #
    #
    # idx_start = 0
    # idx_end = idx_start+240
    #
    # for i, model in enumerate(models):
    #     print(idx_start, idx_end)
    #     y_valid_pred = y_valid_combined_pred[idx_start:idx_end, :, :]
    #     y_valid = y_valid_combined_input[idx_start:idx_end, :, :]
    #     print(np.isnan(y_valid_pred).sum())
    #     y_valid_pred[np.isnan(y_valid)] = np.nan
    #     print(np.isnan(y_valid_pred).sum())
    #
    #     ## If trained on trends
    #     # trend_prediction_2041_2070 = y_valid_pred[0,:,:]
    #     # model_trend_2041_2070 = y_valid[0,:,:]
    #
    #     # eval.plot(trend_prediction_2041_2070, folder_saving, "trend_prediction_on_2041_2070_" + str(model), trend=True)
    #     # eval.plot(model_trend_2041_2070, folder_saving, "trend_model_on_2041_2070_" + str(model), trend=True)
    #     # eval.plot(model_trend_2041_2070 - trend_prediction_2041_2070, folder_saving, "trend_diff_model_and_pred_2041_2070_" + str(model), trend=True)
    #     #
    #     #
    #     #
    #     # model_trend_2041_2070_rms, _ = eval.evaluation_metrics(None,
    #     #                                              model_trend_2041_2070 * 10,
    #     #                                              mask=~np.isnan(model_trend_2041_2070),
    #     #                                              weight_map=weight_map, trend=True)
    #     # trend_prediction_2041_2070_rms, _ = eval.evaluation_metrics(None,
    #     #                                              trend_prediction_2041_2070 * 10,
    #     #                                              mask=~np.isnan(trend_prediction_2041_2070),
    #     #                                              weight_map=weight_map, trend=True)
    #     # diff_model_ml_pred_rms_2041_2070, _ = eval.evaluation_metrics(None,
    #     #                                                         (model_trend_2041_2070 - trend_prediction_2041_2070) * 10,
    #     #                                                         mask=~np.isnan(model_trend_2041_2070),
    #     #                                                         weight_map=weight_map, trend=True)
    #     #
    #     # print("rms of ml prediction 2041_2070 in mm/yr: ", trend_prediction_2041_2070_rms)
    #     # print("rms of diff between clm model and ml prediction 2041_2070 in mm/yr: ", diff_model_ml_pred_rms_2041_2070)
    #     # print("rms of clm model 2041_2070 in mm/yr: ", model_trend_2041_2070_rms)
    #
    #
    #     # trend_prediction_2051_2080 = y_valid_pred[120, :, :]
    #     # model_trend_2051_2080 = y_valid[120, :, :]
    #
    #     ## If trained on averages
    #     points_for_trending = [0, 10 * 12, 20 * 12 - 1]
    #     print(np.nanmax(y_valid_pred[points_for_trending, :, :]), np.nanmin(y_valid_pred[points_for_trending, :, :]), np.nanmax(y_valid[points_for_trending, :, :]), np.nanmin(y_valid[points_for_trending, :, :]))
    #     trend_prediction_2051_2080 = eval.fit_trend(y_valid_pred[points_for_trending, :, :],
    #                                                 ~np.isnan(y_valid[points_for_trending, :, :]), yearly=True,
    #                                                 year_range=range(3))
    #     model_trend_2051_2080 = eval.fit_trend(y_valid[points_for_trending, :, :],
    #                                            ~np.isnan(y_valid[points_for_trending, :, :]), yearly=True,
    #                                            year_range=range(3))
    #
    #
    #
    #     eval.plot(trend_prediction_2051_2080, folder_saving, "trend_prediction_on_2051_2080_" + str(model), trend=True)
    #     eval.plot(model_trend_2051_2080, folder_saving, "trend_model_on_2051_2080_" + str(model), trend=True)
    #     eval.plot(model_trend_2051_2080-trend_prediction_2051_2080, folder_saving, "trend_diff_model_and_pred_on_2051_2080_" + str(model), trend=True)
    #
    #     model_trend_2051_2080_rms, _ = eval.evaluation_metrics(None,
    #                                                        model_trend_2051_2080 * 1000,
    #                                                        mask=~np.isnan(model_trend_2051_2080),
    #                                                        weight_map=weight_map, trend=True)
    #     trend_prediction_2051_2080_rms, _ = eval.evaluation_metrics(None,
    #                                                             trend_prediction_2051_2080 * 1000,
    #                                                             mask=~np.isnan(trend_prediction_2051_2080),
    #                                                             weight_map=weight_map, trend=True)
    #     trend_diff_model_ml_pred_rms_2051_2080, _ = eval.evaluation_metrics(None,
    #                                                                   (
    #                                                                   model_trend_2051_2080 - trend_prediction_2051_2080) * 1000,
    #                                                                   mask=~np.isnan(model_trend_2051_2080),
    #                                                                   weight_map=weight_map, trend=True)
    #
    #     print("rms of ml prediction 2051_2080 in mm/yr: ", trend_prediction_2051_2080_rms)
    #     print("rms of diff between clm model and ml prediction 2051_2080 in mm/yr: ", trend_diff_model_ml_pred_rms_2051_2080)
    #     print("rms of clm model 2051_2080 in mm/yr: ", model_trend_2051_2080_rms)
    #
    #     #
    #     # persistence_trend = X_valid_combined_input[idx_start:idx_end , :, :, 0]
    #     # print(persistence_trend.shape)
    #     # persistence_trend_2041_2070 = persistence_trend[0, :, :]
    #     #
    #     # eval.plot(model_trend_2041_2070 - persistence_trend_2041_2070, folder_saving,
    #     #           "trend_diff_model_and_persistence_2041_2070_" + str(model), trend=True)
    #     #
    #     # diff_model_persistence_rms_2041_2070, _ = eval.evaluation_metrics(None,
    #     #                                                               (
    #     #                                                              model_trend_2041_2070 - persistence_trend_2041_2070) * 10,
    #     #                                                               mask=~np.isnan(model_trend_2041_2070),
    #     #                                                               weight_map=weight_map, trend=True)
    #     #
    #     # print("rms of diff between clm model and persistence 2041_2070 in mm/yr: ", diff_model_persistence_rms_2041_2070)
    #     #
    #     # persistence_trend_2051_2080 = persistence_trend[120, :, :]
    #     #
    #     # eval.plot(model_trend_2051_2080 - persistence_trend_2051_2080, folder_saving,
    #     #           "trend_diff_model_and_persistence_2051_2080_" + str(model), trend=True)
    #     #
    #     # diff_model_persistence_rms_2051_2080, _ = eval.evaluation_metrics(None,
    #     #                                                                   (
    #     #                                                                   model_trend_2051_2080 - persistence_trend_2051_2080) * 10,
    #     #                                                                   mask=~np.isnan(model_trend_2051_2080),
    #     #                                                                   weight_map=weight_map, trend=True)
    #     #
    #     # print("rms of diff between clm model and persistence 2051_2080 in mm/yr: ",
    #     #       diff_model_persistence_rms_2051_2080)
    #
    #     ## Plotting a single prediction map
    #     # prediction_2051_2060 = y_valid_pred[0, :, :]
    #     # prediction_2061_2070 = y_valid_pred[10 * 12, :, :]
    #     #
    #     # model_2051_2060 = y_valid[0, :, :]
    #     # model_2061_2070 = y_valid[10 * 12, :, :]
    #     #
    #     # print(prediction_2051_2060)
    #     # print(np.isnan(model_2051_2060).sum())
    #     # print(np.isnan(model_2061_2070).sum())
    #
    #     # eval.plot(prediction_2051_2060, folder_saving, "avg_prediction_on_2051_2060_" + str(model), trend=True)
    #     # eval.plot(model_2051_2060, folder_saving, "model_on_2051_2060_" + str(model), trend=True)
    #     # eval.plot(model_2051_2060 - prediction_2051_2060, folder_saving,
    #     #           "diff_model_and_pred_on_2051_2060_" + str(model), trend=True)
    #
    #     # model_2051_2060_rms, _ = eval.evaluation_metrics(None,
    #     #                                                  model_2051_2060,
    #     #                                                  mask=~np.isnan(model_2051_2060),
    #     #                                                  weight_map=weight_map, trend=True)
    #     # prediction_2051_2060_rms, _ = eval.evaluation_metrics(None,
    #     #                                                       prediction_2051_2060,
    #     #                                                       mask=~np.isnan(prediction_2051_2060),
    #     #                                                       weight_map=weight_map, trend=True)
    #     # diff_model_ml_pred_rms_2051_2060, _ = eval.evaluation_metrics(None,
    #     #                                                               (
    #     #                                                                       model_2051_2060 - prediction_2051_2060),
    #     #                                                               mask=~np.isnan(model_2051_2060),
    #     #                                                               weight_map=weight_map, trend=True)
    #     #
    #     # print("rms of ml prediction 2051_2060 in meters: ", prediction_2051_2060_rms)
    #     # print("rms of diff between clm model and ml prediction 2051_2060 in meters: ", diff_model_ml_pred_rms_2051_2060)
    #     # print("rms of clm model 2051_2060 in meters: ", model_2051_2060_rms)
    #     #
    #     # eval.plot(prediction_2061_2070, folder_saving, "avg_prediction_on_2061_2070_" + str(model), trend=True)
    #     # eval.plot(model_2061_2070, folder_saving, "model_on_2061_2070_" + str(model), trend=True)
    #     # eval.plot(model_2061_2070 - prediction_2061_2070, folder_saving,
    #     #           "diff_model_and_pred_on_2061_2070_" + str(model), trend=True)
    #
    #     # model_2061_2070_rms, _ = eval.evaluation_metrics(None,
    #     #                                                  model_2061_2070,
    #     #                                                  mask=~np.isnan(model_2061_2070),
    #     #                                                  weight_map=weight_map, trend=True)
    #     # prediction_2061_2070_rms, _ = eval.evaluation_metrics(None,
    #     #                                                       prediction_2061_2070,
    #     #                                                       mask=~np.isnan(prediction_2061_2070),
    #     #                                                       weight_map=weight_map, trend=True)
    #     # diff_model_ml_pred_rms_2061_2070, _ = eval.evaluation_metrics(None,
    #     #                                                               (
    #     #                                                                       model_2061_2070 - prediction_2061_2070),
    #     #                                                               mask=~np.isnan(model_2061_2070),
    #     #                                                               weight_map=weight_map, trend=True)
    #     #
    #     # print("rms of ml prediction 2061_2070 in meters: ", prediction_2061_2070_rms)
    #     # print("rms of diff between clm model and ml prediction 2061_2070 in meters: ", diff_model_ml_pred_rms_2061_2070)
    #     # print("rms of clm model 2061_2070 in meters: ", model_2061_2070_rms)
    #
    #     idx_start = idx_end
    #     idx_end = idx_start+240
    #




if __name__=='__main__':
    main()


