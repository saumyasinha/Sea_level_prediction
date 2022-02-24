import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
from ModuleLearning import preprocessing, eval
from ModuleLearning.ModuleCNN import train as train_cnn

path_local = "/Users/saumya/Desktop/Sealevelrise/"
path_cluster = "/pl/active/machinelearning/Saumya/ML_for_sea_level/"
path_project = path_local

path_data = path_project+"Data/"
path_models = path_project+"ML_Models/"
path_data_fr = path_data + "Forced_Responses/"

## which climate model to work on
models = ['CESM1LE'] # ['MIROC-ES2L'] #['CESM2LE'] #['MPI-ESM1-2-HR']

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
model_type = "ConvLSTM" #"DilatedUnet3d"#"Unet"#"SmaAT_Unet" #"DilatedUnet"#"Unet_Attn" #"ConvLSTM" #

## if we want to have probabilsitic prediction
quantile = False
alphas = np.arange(0.05, 1.0, 0.05)
q50 = 9

## folders to finally save the model
reg = "CNN/Unet" #convLSTM/"# Unet"
sub_reg = "cnn_with_more_hidden_1yr_lag_dilated_convlstm_downscaled_weighted_changed_years_not_normalized"#"
# sub_reg = "cnn_with_2yrs_lag_large_batchnorm_unet3d_downscaled_weighted_changed_years_not_normalized"#"final_cnn_with_1yr_lag_large_batchnorm_unet_downscaled_weighted_changed_years_not_normalized"


## Hyperparameters
hidden_dim = 25 #25
num_layers=1
kernel_size = (3,3)

batch_size = 6
epochs = 1#200
lr = 1e-4

features = ["sea_level"]
n_features = len(features)
n_prev_months = 12 ##seq-length of the deep sequence models

yearly = False  #in case you want to look at year level data instead of month level
downscaling = True #converting 360*180 to 180*90
include_patches = False
include_heat = False #include the heat content feature
attention = False


if include_heat:
    features.extend(["heat_content"])



def main():

    for model in models:

        folder_saving = path_models+ model + "/" + reg + "/"+ sub_reg + "/"
        os.makedirs(
            folder_saving, exist_ok=True)

        f = open(folder_saving +"/results.txt", 'a')

        ## these weights are cosine of latitude, important for using weighted RMSE as a loss function
        weight_map = np.load(historical_path+"weights_historical_"+model+"_zos_fr_1850_2014.npy")
        weight_map = np.abs(weight_map)
        if downscaling:
            weight_map = block_reduce(weight_map, (2,2), np.mean) #(2,2)
        print(weight_map.shape, np.max(weight_map), np.min(weight_map))

        train, test = preprocessing.create_train_test_split(model, historical_path, future_path, train_start_year, train_end_year, test_start_year, test_end_year, n_prev_months, lead_years, downscaling)
        # np.save(path_data_fr+ model + "/"+"train_for_"+str(train_start_year)+"-"+str(train_end_year)+".npy", train)
        # np.save(path_data_fr+ model + "/"+"test_for_" + str(test_start_year) + "-" + str(test_end_year) + ".npy", test)

        ## converting cms to meters
        train = train / 100
        test = test / 100

        ## bulding the X,y dataset by assigning the predictions for every t
        X_train, y_train, X_test, y_test = preprocessing.create_labels(train, test, lead_years,n_prev_months)

        if include_heat:
            train_heat, test_heat = preprocessing.create_train_test_split(model, historical_heat_path, future_heat_path,
                                                                          train_start_year,
                                                                          train_end_year, test_start_year,
                                                                          test_end_year,
                                                                          n_prev_months, lead_years, downscaling, heat=True)

            train_heat = train_heat / 100
            test_heat = test_heat / 100

            print("max of heat",np.max(train_heat), np.max(test_heat))

            X_train_heat, y_train_heat, X_test_heat, y_test_heat = preprocessing.create_labels(train_heat, test_heat, lead_years, n_prev_months)

            X_train = np.stack([X_train, X_train_heat], axis=3)
            X_test = np.stack([X_test, X_test_heat], axis=3)

            print("if including heat: ", X_train.shape, X_test.shape)


        ## splitting train into train+valid
        split_index = 30 if yearly else 12*30

        ##if working with years instead of months
        if yearly:
            n_prev_times = int(n_prev_months/12)
            X_train, y_train, X_test, y_test = preprocessing.convert_month_to_years(X_train),preprocessing.convert_month_to_years(y_train),preprocessing.convert_month_to_years(X_test),preprocessing.convert_month_to_years(y_test)
        else:
            n_prev_times = n_prev_months
            # X_train, X_test, y_train, y_test = preprocessing.normalize_from_train(X_train, X_test, y_train, y_test, split_index)

        ## remove land values
        X_train = preprocessing.remove_land_values(X_train)
        X_test = preprocessing.remove_land_values(X_test)

        ## add previous timestep values
        X_train = preprocessing.include_prev_timesteps(X_train, n_prev_times, include_heat)
        X_test = preprocessing.include_prev_timesteps(X_test, n_prev_times, include_heat)

        y_train = y_train[:,:,n_prev_times:]
        y_test = y_test[:,:,n_prev_times:]

        y_train = np.transpose(y_train, (2,0,1)) #y_train.reshape(-1,lon,lat)
        y_test = np.transpose(y_test, (2,0,1)) #y_test.reshape(-1, lon, lat)


        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print(np.max(X_train),np.max(y_train))

        # X_train, X_valid, y_train, y_valid = train_test_split(
        #     X_train, y_train, test_size=0.2, random_state=42)


        train_valid_split_index = len(X_train) - split_index #keeping later 30 years for validation
        X,y = X_train,y_train
        X_train = X[:train_valid_split_index]
        y_train = y[:train_valid_split_index]
        X_valid = X[train_valid_split_index:]
        y_valid = y[train_valid_split_index:]

        # print(weight_map_train.shape, weight_map_valid.shape)

        print("train/valid sizes: ", len(X_train), " ", len(X_valid))

        # X_train, X_valid, X_test = preprocessing.normalize_from_train(X_train,X_valid, X_test)

        # weight_map_train = np.repeat(weight_map[None, ...], len(X_train), axis=0)
        # weight_map_valid = np.repeat(weight_map[None, ...], len(X_valid), axis=0)
        X_train_input, y_train_input = X_train,y_train
        X_valid_input, y_valid_input = X_valid, y_valid
        X_test_input, y_test_input = X_test, y_test
        # weight_map_train_input = weight_map_train
        # weight_map_valid_input = weight_map_valid


        if include_patches:
            X_train_input,y_train_input = preprocessing.get_image_patches(X_train,y_train)# preprocessing.downscale_input(X_train,y_train)
            X_valid_input, y_valid_input =  preprocessing.get_image_patches(X_valid, y_valid) #preprocessing.downscale_input(X_valid, y_valid)
            X_test_input, y_test_input = preprocessing.get_image_patches(X_test, y_test) #preprocessing.downscale_input(X_test, y_test) # #
            # weight_map_train_input = preprocessing.get_image_patches(None,weight_map_train)
            # weight_map_valid_input = preprocessing.get_image_patches(None, weight_map_valid)



        model_saved = "model_at_lead_"+str(lead_years)+"_yrs"
        train_cnn.basic_CNN_train(X_train_input, y_train_input, X_valid_input, y_valid_input, weight_map, n_features,  n_prev_times+1, epochs, batch_size, lr, folder_saving, model_saved, include_heat, quantile, alphas, model_type = model_type, hidden_dim = hidden_dim, num_layers = num_layers, kernel_size=kernel_size, attention = attention)
        # valid_rmse, valid_mae, test_rmse, test_mae, valid_mask, test_mask = train_cnn.basic_CNN_test(X_train_input, X_valid_input, y_valid_input, X_test_input, y_test_input, weight_map, n_features, n_prev_times+1, folder_saving, model_saved, quantile, alphas, model_type = model_type, hidden_dim = hidden_dim, num_layers = num_layers, kernel_size=kernel_size, attention=attention)
        # f.write('\n evaluation metrics (rmse, mae) on valid data ' + str(valid_rmse) + "," + str(valid_mae) +'\n')
        # f.write('\n evaluation metrics (rmse, mae) on test data ' + str(test_rmse) + "," + str(test_mae) + '\n')
        # f.close()
        #
        #
        # #####Visualizations####################
        # #### get trend plots######
        # y_valid_pred = np.load(folder_saving+"/valid_predictions.npy")
        # # # # # print(y_valid_pred.shape)
        # # # #
        # y_valid_wo_patches, valid_mask = train_cnn.get_target_mask(y_valid)
        # #
        # valid_trend = eval.fit_trend(y_valid_pred, valid_mask, yearly=yearly)
        # eval.plot(valid_trend, folder_saving, "valid_trend_2041-2070_same_yaxis", trend=True)
        # model_trend = eval.fit_trend(y_valid, valid_mask, yearly=yearly)
        # eval.plot(model_trend, folder_saving, "model_trend_2041-2070_same_y_axis", trend=True)
        # # diff = model_trend - valid_trend
        # # eval.plot(diff, folder_saving, "diff_wihtout_dots_trend_2041-2070_same_y_axis", trend=True)
        # # eval.plot(model_trend/np.abs(diff), folder_saving, "signal_to_noise_trend_2041-2070_same_y_axis", trend=True)
        # rmse_trend, mae_trend = eval.evaluation_metrics(model_trend*1000, valid_trend*1000, mask = ~np.isnan(valid_trend), weight_map=weight_map, trend=True)
        #
        # print("rmse and log rmse of the trend plots on validation is: ", rmse_trend, np.log(rmse_trend))
        # # #For unet downscaled: rmse and log rmse of the trend plots on validation is:  0.654340228637174 -0.42412783552226774

        ### plot true vs predicitons on best/worst rmse pts
        # mean_for_valid_period = np.mean(y_valid, axis=0)
        # y_valid_mean_removed = y_valid - mean_for_valid_period[np.newaxis, :, :]
        # mean_pred_for_valid_pred = np.mean(y_valid_pred, axis=0)
        # y_valid_pred_mean_removed = y_valid_pred - mean_pred_for_valid_pred[np.newaxis, :, :]
        # #
        # # # y_valid_mean = np.mean(y_valid, axis=0)
        # # # # print(np.isnan(y_valid_mean).sum())
        # # # # y_valid_pred_mean = np.mean(y_valid_pred, axis=0)
        # # # # weighted_diff2 = ((y_valid_mean - y_valid_pred_mean)**2) * weight_map
        # weighted_diff2 = (diff**2)*weight_map
        # weighted_diff2 = np.ma.masked_where(np.isnan(weighted_diff2), weighted_diff2)
        #
        # sorted_points_wrt_error = np.dstack(np.unravel_index(weighted_diff2.argsort(axis=None), weighted_diff2.shape))
        # print(sorted_points_wrt_error, sorted_points_wrt_error.shape)
        #
        # # print(np.unravel_index(np.nanargmin(weighted_diff2), weighted_diff2.shape), np.unravel_index(np.nanargmax(weighted_diff2),weighted_diff2.shape))
        # best_counter = 0
        # worst_counter = -1
        # count=0
        # lats = np.load(historical_path+"/latitudes.npy")
        # lats = block_reduce(lats, (2,), np.mean)
        # lons = np.load(historical_path + "/longitudes.npy")
        # lons = block_reduce(lons, (2,), np.mean)
        #
        # while count<10:
        #     if count<5:
        #         pt = sorted_points_wrt_error[0,best_counter,:]
        #         best_counter = best_counter + 1
        #         if ~np.isnan(np.sum(y_valid_mean_removed[:,pt[0],pt[1]])):
        #             print(pt)
        #             print(lons[pt[0]],lats[pt[1]])
        #             eval.single_point_test(pt[0], pt[1], y_valid_pred_mean_removed, y_valid_mean_removed, years = list(range(2041,2071)), count=count, folder_saving=folder_saving)
        #             count=count+1
        #
        #
        #     else:
        #         pt = sorted_points_wrt_error[0,worst_counter,:]
        #         worst_counter = worst_counter-1
        #         if ~np.isnan(np.sum(y_valid_mean_removed[:,pt[0],pt[1]])):
        #             print(pt)
        #             print(lons[pt[0]], lats[pt[1]])
        #             eval.single_point_test(pt[0], pt[1], y_valid_pred_mean_removed, y_valid_mean_removed, years = list(range(2041,2071)), count=count, folder_saving=folder_saving)
        #             count=count+1

        # # ## plot persistence
        # y_persistence = y_train[-30*12:,:,:]
        # persistence_trend = eval.fit_trend(y_persistence, valid_mask, yearly=yearly)
        # # eval.plot(persistence_trend, folder_saving, "persistence_trend_2041-2070_same_yaxis", trend=True)
        # model_trend = eval.fit_trend(y_valid, valid_mask, yearly=yearly)
        # eval.plot(model_trend - persistence_trend, folder_saving, "diff_model_and_persis_trend_2041-2070_same_y_axis", trend=True)

        # persistence_rmse, persistence_mae = eval.evaluation_metrics(model_trend*1000, persistence_trend*1000, mask = ~np.isnan(persistence_trend), weight_map=weight_map)
        # persistence_rmse, persistence_mae = eval.evaluation_metrics(y_valid, y_persistence,
        #                                                             mask=valid_mask,
        #                                                             weight_map=weight_map)
        # #
        # print("persistence trend rmse, mae: ",persistence_rmse, persistence_mae)
        # # persistence trend rmse, mae: 0.6830641514406547 , 0.47213281289195413
        #persistence model (and not trend) rmse, mae:  0.0363006390941568 0.026479292738289497
        #persistence  model (for test) rmse, mae: 0.04712170069153805 0.033402950851230594








if __name__=='__main__':
    main()


