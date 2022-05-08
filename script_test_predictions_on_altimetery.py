import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
from ModuleLearning import preprocessing, eval
from ModuleLearning.ModuleCNN import train as train_cnn
import matplotlib.pyplot as plt

path_local = "/Users/saumya/Desktop/Sealevelrise/"
path_cluster = "/pl/active/machinelearning/Saumya/ML_for_sea_level/"
path_project = path_local

path_data = path_project+"Data/"
path_data_obs = path_data + "Observations/"
path_models = path_project+"ML_Models/"
path_data_fr = path_data + "Forced_Responses/"

## which climate model to work on
models = ['CESM1LE','CESM2LE'] # ['MIROC-ES2L'] #['MPI-ESM1-2-HR']

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

lead_years = 30
model_type = "DilatedUnet"#"Unet"#"SmaAT_Unet" #"DilatedUnet"#"Unet_Attn" #"ConvLSTM" #

## if we want to have probabilsitic prediction
quantile = False
alphas = np.arange(0.05, 1.0, 0.05)
q50 = 9

## folders to finally save the model
reg = "CNN/trend_Unet/"
# sub_reg = "cnn_with_1yr_lag_convlstm_downscaled_weighted_changed_years_not_normalized"
# sub_reg = "_rerun_cnn_with_1yr_lag_large_batchnorm_unet_downscaled_weighted_changed_years_not_normalized"#"final_cnn_with_1yr_lag_convlstm_downscaled_weighted_changed_years_not_normalized"#"
# sub_reg = "_trend_cnn_with_0lag_small_batchnorm_unet_weight_decay1e-4_downscaled_weighted_changed_years_not_normalized"
sub_reg = "_trend_cnn_with_0lag_batchnorm_unetdilated_weight_decay1e-6_downscaled_weighted_changed_years_not_normalized"
## Hyperparameters
hidden_dim = 12
num_layers=1
kernel_size = [(3,3)]

batch_size = 6
epochs = 200#200
lr = 1e-4

features = ["sea_level"]
n_features = len(features)
n_prev_months = 0 ##seq-length of the deep sequence models

yearly = False  #in case you want to look at year level data instead of month level
downscaling = True #converting 360*180 to 180*90
include_patches = False
include_heat = False #include the heat content feature
attention=False

if include_heat:
    features.extend(["heat_content"])



def main():



    weight_map = np.load(historical_path + "weights_historical_CESM1LE_zos_fr_1850_2014.npy")
    weight_map = np.abs(weight_map)
    if downscaling:
        weight_map = block_reduce(weight_map, (2, 2), np.mean)  # (2,2)
    print(weight_map.shape, np.max(weight_map), np.min(weight_map))

    if len(models) > 1:
        folder_saving = path_models + "combined_CESM1and2/" + reg + "/" + sub_reg + "/"
    else:
        folder_saving = path_models + models[0] + "/" + reg + "/" + sub_reg + "/"

    os.makedirs(
        folder_saving, exist_ok=True)

    f = open(folder_saving + "/results.txt", 'a')


    X_test = np.load(path_data_obs+'npy_files/altimeter2deg.npy')
    X_test = X_test[ :, :,:-5] ## keeping only the altimeter data till 2019 (1993-2019)

    print(X_test.shape)
    X_test[X_test == 1e+36] = np.nan
    # if downscaling:
    #     X_test = preprocessing.downscale_input(X_test)

    X_test = X_test/10 #converting to cms
    print(X_test.shape)

    # altimeter_SSH = X_test[:, :, n_prev_times:].copy()
    altimeter_SSH = X_test.copy()

    mask_ssh = ~np.isnan(X_test)
    altimeter_trend = eval.fit_trend(np.transpose(altimeter_SSH, (2, 0, 1)), np.transpose(mask_ssh, (2, 0, 1)), yearly=False, year_range=range(1993,2020))
    print(altimeter_trend.shape)
    X_test = altimeter_trend[np.newaxis,:,:].copy()

    print(X_test.shape)
    ## remove land values
    X_test = preprocessing.remove_land_values(X_test)

    ## add previous timestep values
    # X_test = preprocessing.include_prev_timesteps(X_test, n_prev_times, include_heat)
    X_test = X_test[:,:,:,np.newaxis]
    print(X_test.shape, np.max(X_test), np.min(X_test))

    model_saved = "model_at_lead_"+str(lead_years)+"_yrs"

    n_prev_times = n_prev_months

    altimeter_prediction_list_over_multiple_runs = []
    for iter in range(3):
    # train_cnn.basic_CNN_train(X_train_input, y_train_input, X_valid_input, y_valid_input, weight_map, n_features,  n_prev_times+1, epochs, batch_size, lr, folder_saving, model_saved, include_heat, quantile, alphas, model_type = model_type, hidden_dim = hidden_dim, num_layers = num_layers, kernel_size=kernel_size)
    #     valid_rmse, valid_mae, test_rmse, test_mae, valid_mask, test_mask = train_cnn.basic_CNN_test(None, None, None, None, X_test, None, weight_map, n_features, n_prev_times+1, folder_saving+"/"+str(iter)+"/", model_saved, quantile, alphas, model_type = model_type, hidden_dim = hidden_dim, num_layers = num_layers, kernel_size=kernel_size, attention=attention)
    # f.write('\n evaluation metrics (rmse, mae) on valid data ' + str(valid_rmse) + "," + str(valid_mae) +'\n')
    # f.write('\n evaluation metrics (rmse, mae) on altimeter data ' + str(test_rmse) + "," + str(test_mae) + '\n')
    # f.close()
        prediction_trend_per_run = np.load(folder_saving + "/"+str(iter) + "/altimeter_predictions.npy")
        print(iter)
        print(prediction_trend_per_run.shape)
        altimeter_prediction_list_over_multiple_runs.append(prediction_trend_per_run)


    #####Visualizations####################
    #### get trend plots######
    print(np.stack(altimeter_prediction_list_over_multiple_runs,axis=0).shape)
    prediction_trend = np.mean(np.stack(altimeter_prediction_list_over_multiple_runs,axis=0),axis=0)

    print(prediction_trend.shape)
    print(np.isnan(prediction_trend).sum())

    prediction_trend[np.isnan(altimeter_trend)] = np.nan
    print(np.isnan(prediction_trend).sum())

    np.save("DilatedUnet_trained_on_CESM1and2_trends_for_2023-2049",prediction_trend)
    # y_test_pred = y_test_pred[:-5, :, :]
    # print(y_test_pred.shape)
    # mask = mask[:-5,:,:]
    # print(mask.shape, mask.sum())
    # eval.plot(prediction_trend, folder_saving, "altimeter_prediction_trend_2023-2049", trend=True)
    # # altimeter_data = altimeter_data[:-5, :, :]
    # # print(np.nanmin(altimeter_data), np.nanmax(altimeter_data)) #-2.2653828 1.6158351
    # # altimeter_trend = eval.fit_trend(altimeter_data, mask, yearly=yearly, year_range = range(1994,2020))
    # eval.plot(altimeter_trend, folder_saving, "altimeter_trend_1993-2019", trend=True)
    #
    # # eval.plot(prediction_trend - altimeter_trend, folder_saving, "altimeter_prediction_trend-altimeter_trend", trend=True)
    # altimeter_pred_rms,_ = eval.evaluation_metrics(None,prediction_trend*10, mask = ~np.isnan(prediction_trend), weight_map=weight_map, trend=True)
    # altimeter_rms, _ = eval.evaluation_metrics(None, altimeter_trend*10, mask=~np.isnan(altimeter_trend),
    #                                                 weight_map=weight_map, trend=True)
    # diff_ML_pred_altimeter_rms, _ = eval.evaluation_metrics(None, (prediction_trend-altimeter_trend) * 1000, mask=~np.isnan(altimeter_trend),
    #                                            weight_map=weight_map, trend=True)
    #### get trend plots on climate model for comparison ####
    # climate_model_2024_2049 = np.load(folder_saving+"/true_climate_model_2024-2049.npy")
    # # print(np.isnan(climate_model_2024_2049).sum())
    # _, climate_mask = train_cnn.get_target_mask(climate_model_2024_2049)
    # # # #
    # climate_model_2024_2049_trend = eval.fit_trend(climate_model_2024_2049, climate_mask, yearly=yearly,
    #                                                    year_range=range(2024, 2050))
    #
    # climate_model_1994_2019 = np.load(folder_saving+"/climate_model_1994_2019.npy")
    # # eval.learn_map_climate_model_to_altimeter(climate_model_1994_2019,altimeter_data, weight_map, folder_saving)
    # # print(type(climate_model_1994_2019))
    # climate_model_persistence_trend = eval.fit_trend(climate_model_1994_2019, climate_mask, yearly=yearly, year_range = range(1994,2020))
    # eval.plot(climate_model_persistence_trend, folder_saving, "climate_model_persistence_trend", trend=True)
    #
    # climate_model_MLpredictions = np.load(folder_saving+"/predictions_on_climate_model_2024-2049.npy")
    # climate_model_MLpredictions_trend = eval.fit_trend(climate_model_MLpredictions, climate_mask, yearly=yearly,
    #                                                  year_range=range(2024, 2050))
    #
    # ML_pred_on_clm_model_rms, _ = eval.evaluation_metrics(None, climate_model_MLpredictions_trend * 1000, mask=~np.isnan(climate_model_MLpredictions_trend),
    #                                                 weight_map=weight_map, trend=True)
    # clm_model_persistence_rms, _ = eval.evaluation_metrics(None, climate_model_persistence_trend * 1000, mask=~np.isnan(climate_model_persistence_trend),
    #                                            weight_map=weight_map, trend=True)
    # diff_ml_pred_clm_model_rms,_ = eval.evaluation_metrics(None, (climate_model_MLpredictions_trend-climate_model_2024_2049_trend)* 1000, mask=~np.isnan(climate_model_MLpredictions_trend),
    #                                            weight_map=weight_map, trend=True)
    # #
    print("rms of altimeter in mm/yr: ", altimeter_rms)
    print("rms of altimeter predictions in mm/yr: ", altimeter_pred_rms)
    # print("rms of altimeter predictions - altimeter in mm/yr: ", diff_ML_pred_altimeter_rms)
    #
    # print("rms of clm persistence in mm/yr: ", clm_model_persistence_rms)
    # print("rms of ml predictions on clm model in mm/yr: ", ML_pred_on_clm_model_rms)
    # print("rms of ml predictions - true clm model in mm/yr: ", diff_ml_pred_clm_model_rms)



    # climate_model_MLpredictions_scaled_2024_2049 = eval.post_process_climate_to_altimeter(climate_model_MLpredictions, folder_saving)
    # np.save(folder_saving+"climate_model_MLpredictions_scaled_2024_2049.npy", climate_model_MLpredictions_scaled_2024_2049)

    # climate_model_MLpredictions_scaled_trend = eval.fit_trend(climate_model_MLpredictions_scaled_2024_2049, mask, yearly=yearly,
    #                                          year_range=range(2024, 2050))
    # eval.plot(climate_model_MLpredictions_scaled_trend, folder_saving, "climate_model_MLpredictions_scaled_trend", trend=True)
    #
    # eval.plot(prediction_trend-climate_model_MLpredictions_scaled_trend, folder_saving, "altimer_prediction_trend-clm_model_postporocessed_trend_2024-2049", trend=True)

    # eval.plot(climate_model_MLpredictions_trend-climate_model_2024_2049_trend, folder_saving, "climate_model_MLpredictions_trend-climate_model_2024-2049", trend=True)


    # rmse, mae = eval.evaluation_metrics(climate_model_2024_2049, climate_model_MLpredictions,
    #                                                             mask=climate_mask,
    #                                                             weight_map=weight_map)

    ## Note
    # We are making a prediction from 1994+30yrs (not 1993)





if __name__=='__main__':
    main()

## A plot to just figure out relation between  altimeter and climate model trend
# plt.scatter(climate_model_persistence_trend*1000, altimeter_trend*1000)
# plt.ylim([-10, 20])
# plt.ylabel("Altimeter trend in mm/yr")
# plt.xlabel("Climate model trend in mm/yr")
# plt.title("scatter plot of altimeter and climate model trend for 1994-2019")
# plt.savefig("comparing_alt_vs_clm_model_trend")