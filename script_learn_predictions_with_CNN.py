import os
import numpy as np
from sklearn.model_selection import train_test_split
# from Sea_level_prediction.ModuleLearning import preprocessing,eval
# from Sea_level_prediction.ModuleLearning.ModuleCNN import train as train_cnn
from skimage.measure import block_reduce
from ModuleLearning import preprocessing, eval
from ModuleLearning.ModuleCNN import train as train_cnn

path_local = "/Users/saumya/Desktop/Sealevelrise/"
path_cluster = "/pl/active/machinelearning/ML_for_sea_level/"
path_project = path_cluster
path_data = path_project+"Data/"
path_models = path_project+"ML_Models/"
path_data_fr = path_data + "Forced_Responses/"

models = ['CESM1LE'] # ['MIROC-ES2L'] #['CESM2LE'] #['MPI-ESM1-2-HR']  #

path_sealevel_folder = path_data_fr + "zos/"
path_heatcontent_folder = path_data_fr + "heatfull/"

path_folder = path_sealevel_folder

historical_path = path_folder + "1850-2014/npy_files/"
future_path = path_folder + "2015-2100/npy_files/"

train_start_year = 1930 # 1880 #
train_end_year =  2040 #1990 #
test_start_year =  2041 #1991
test_end_year =  2070 #2020 #

lead_years = 30
quantile = False
convlstm = True
hidden_dim = 15
num_layers=1
alphas = np.arange(0.05, 1.0, 0.05)
q50 = 9
reg = "CNN"



# sub_reg = "cnn_with_1yr_lag_unet_w_patches_not_normalized"
# sub_reg = "cnn_with_1yr_lag_unet_with_patches_weighted_changed_years_not_normalized"
sub_reg = "cnn_with_1yr_lag_convlstm_downscaled_weighted_changed_years_not_normalized"

## Hyperparameters
features = ["sea_level"]
n_features = len(features)
n_prev_months = 12 #12
yearly = False
downscaling = True



batch_size = 4
epochs = 200
lr = 1e-4



def main():

    for model in models:

        folder_saving = path_models+ model + "/" + reg + "/"+ sub_reg + "/"
        os.makedirs(
            folder_saving, exist_ok=True)

        f = open(folder_saving +"/results.txt", 'a')

        weight_map = np.load(historical_path+"weights_historical_"+model+"_zos_fr_1850_2014.npy")
        weight_map = np.abs(weight_map)
        if downscaling:
            weight_map = block_reduce(weight_map, (2,2), np.mean)
        print(weight_map.shape, np.max(weight_map), np.min(weight_map))

        train, test = preprocessing.create_train_test_split(model, historical_path, future_path, train_start_year, train_end_year, test_start_year, test_end_year, n_prev_months, lead_years, downscaling)
        # np.save(path_data_fr+ model + "/"+"train_for_"+str(train_start_year)+"-"+str(train_end_year)+".npy", train)
        # np.save(path_data_fr+ model + "/"+"test_for_" + str(test_start_year) + "-" + str(test_end_year) + ".npy", test)

        train = train / 100
        test = test / 100

        X_train, y_train, X_test, y_test = preprocessing.create_labels(train, test, lead_years,n_prev_months)

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
        X_train = preprocessing.include_prev_timesteps(X_train, n_prev_times)
        X_test = preprocessing.include_prev_timesteps(X_test, n_prev_times)

        y_train = y_train[:,:,n_prev_times:]
        y_test = y_test[:,:,n_prev_times:]

        y_train = np.transpose(y_train, (2,0,1)) #y_train.reshape(-1,lon,lat)
        y_test = np.transpose(y_test, (2,0,1)) #y_test.reshape(-1, lon, lat)

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print(np.max(X_train),np.max(y_train))

        # X_train, X_valid, y_train, y_valid = train_test_split(
        #     X_train, y_train, test_size=0.2, random_state=42)


        train_valid_split_index = len(X_train) - split_index #2*120 #keeping later 10 years for validation
        X,y = X_train,y_train
        X_train = X[:train_valid_split_index]
        y_train = y[:train_valid_split_index]
        X_valid = X[train_valid_split_index:]
        y_valid = y[train_valid_split_index:]

        # print(weight_map_train.shape, weight_map_valid.shape)

        print("train/valid sizes: ", len(X_train), " ", len(X_valid))

        # X_train, X_valid, X_test = preprocessing.normalize_from_train(X_train,X_valid, X_test)

        weight_map_train = np.repeat(weight_map[None, ...], len(X_train), axis=0)
        weight_map_valid = np.repeat(weight_map[None, ...], len(X_valid), axis=0)
        X_train_input, y_train_input = X_train,y_train
        X_valid_input, y_valid_input = X_valid, y_valid
        X_test_input, y_test_input = X_test, y_test
        weight_map_train_input = weight_map_train
        weight_map_valid_input = weight_map_valid

        if downscaling is False:
            X_train_input,y_train_input = preprocessing.get_image_patches(X_train,y_train)# preprocessing.downscale_input(X_train,y_train)
            X_valid_input, y_valid_input =  preprocessing.get_image_patches(X_valid, y_valid) #preprocessing.downscale_input(X_valid, y_valid)
            X_test_input, y_test_input = preprocessing.get_image_patches(X_test, y_test) #preprocessing.downscale_input(X_test, y_test) # #
            weight_map_train_input = preprocessing.get_image_patches(None,weight_map_train)
            weight_map_valid_input = preprocessing.get_image_patches(None, weight_map_valid)
            print(weight_map_train_input.shape)




        model_saved = "model_at_lead_"+str(lead_years)+"_yrs"

        y_valid_input_copy = y_valid_input.copy()  # if you are not doing this then pass X_valid and y_valid as None
        # y_valid_copy = y_valid.copy()
        train_cnn.basic_CNN_train(X_train_input, y_train_input, X_valid_input, y_valid_input, weight_map_train_input, weight_map_valid_input, n_features,  n_prev_times+1, epochs, batch_size, lr, folder_saving, model_saved, quantile, alphas, convlstm=convlstm, hidden_dim = hidden_dim, num_layers = num_layers)
        valid_rmse, valid_mae, test_rmse, test_mae, valid_mask, test_mask = train_cnn.basic_CNN_test(X_valid_input, y_valid_input_copy, X_test_input, y_test_input, weight_map, n_features, n_prev_times+1, folder_saving, model_saved, quantile, alphas,convlstm=convlstm, hidden_dim = hidden_dim, num_layers = num_layers)
        f.write('\n evaluation metrics (rmse, mae) on valid data ' + str(valid_rmse) + "," + str(valid_mae) +'\n')
        f.write('\n evaluation metrics (rmse, mae) on test data ' + str(test_rmse) + "," + str(test_mae) + '\n')
        f.close()
        #
        # y_valid_pred = np.load(folder_saving+"/valid_predictions.npy")
        # print(y_valid_pred.shape)
        #
        # # #
        # y_valid_wo_patches, valid_mask = train_cnn.get_target_mask(y_valid)
        # valid_trend = eval.fit_trend(y_valid_pred, valid_mask, yearly=yearly)
        # eval.plot(valid_trend, folder_saving, "valid_trend_1991-2020_same_yaxis", trend=True)
        # model_trend = eval.fit_trend(y_valid, valid_mask, yearly=yearly)
        # eval.plot(model_trend, folder_saving, "model_trend_1991-2020_same_y_axis", trend=True)
        # eval.plot(model_trend - valid_trend, folder_saving, "diff_trend_1991-2020_same_y_axis", trend=True)

        # # # #
        # # yr_JAN2014 = y_valid[-7*12]
        # # yr_JAN2014_pred = y_valid_pred[-7*12]
        # # eval.plot(yr_JAN2014,folder_saving, "model_JAN2014_sla", index = -12)
        # # eval.plot(yr_JAN2014_pred, folder_saving,"predicted_JAN2014_sla", index = -12)
        # # eval.plot(yr_JAN2014 - yr_JAN2014_pred, folder_saving, "model-predicted_JAN2014_sla", index=-12)
        # #
        # # yr_DEC2014 = y_valid[(-7 * 12) +11]
        # # yr_DEC2014_pred = y_valid_pred[(-7 * 12) +11]
        # # eval.plot(yr_DEC2014, folder_saving, "model_DEC2014_sla", index = -1)
        # # eval.plot(yr_DEC2014_pred, folder_saving, "predicted_DEC2014_sla", index = -1)
        # # eval.plot(yr_DEC2014 - yr_DEC2014_pred, folder_saving, "model-predicted_DEC2014_sla", index=-1)
        # #
        # # yr_JAN2010 = y_valid[-11 * 12]
        # # yr_JAN2010_pred = y_valid_pred[-11 * 12]
        # # eval.plot(yr_JAN2010, folder_saving, "model_JAN2010_sla", index = -5*12)
        # # eval.plot(yr_JAN2010_pred, folder_saving, "predicted_JAN2010_sla", index = -5*12)
        # # eval.plot(yr_JAN2010 - yr_JAN2010_pred, folder_saving, "model-predicted_JAN2010_sla", index=-5 * 12)
        #
        # # yr_DEC2010 = y_valid[(-11 * 12) + 11]
        # # yr_DEC2010_pred = y_valid_pred[(-11 * 12) + 11]
        # # # eval.plot(yr_DEC2010, folder_saving, "model_DEC2010_sla", index = (-5*12)+11)
        # # # eval.plot(yr_DEC2010_pred, folder_saving, "predicted_DEC2010_sla", index = (-5*12)+11)
        # # eval.plot(yr_DEC2010 - yr_DEC2010_pred, folder_saving, "model-predicted_DEC2010_sla", index=(-5 * 12) + 11)
        # # #
        # # yr_JAN2004 = y_valid[-17 * 12]
        # # yr_JAN2004_pred = y_valid_pred[-17 * 12]
        # # # eval.plot(yr_JAN2004, folder_saving, "model_JAN2004_sla", index = -11*12)
        # # # eval.plot(yr_JAN2004_pred, folder_saving, "predicted_JAN2004_sla", index = -11*12)
        # # eval.plot(yr_JAN2004 - yr_JAN2004_pred, folder_saving, "model-predicted_JAN2004_sla", index=-11 * 12)
        # #
        # # # yr_DEC2004 = y_valid[(-17 * 12)+11]
        # # # yr_DEC2004_pred = y_valid_pred[(-17 * 12)+11]
        # # # eval.plot(yr_DEC2004, folder_saving, "model_DEC2004_sla", index = (-11*12)+11)
        # # # eval.plot(yr_DEC2004_pred, folder_saving, "predicted_DEC2004_sla", index =  (-11*12)+11)
        # # # eval.plot(yr_DEC2004 - yr_DEC2004_pred, folder_saving, "model-predicted_DEC2004_sla", index=(-11 * 12) + 11)
        # # #
        # # yr_JAN1994 = y_valid[-27 * 12]
        # # yr_JAN1994_pred = y_valid_pred[-27 * 12]
        # # # eval.plot(yr_JAN1994, folder_saving, "model_JAN1994_sla", index = -21*12)
        # # # eval.plot(yr_JAN1994_pred, folder_saving, "predicted_JAN1994_sla", index = -21*12)
        # # eval.plot(yr_JAN1994 - yr_JAN1994_pred, folder_saving, "model-predicted_JAN1994_sla", index=-21 * 12)
        #
        #
        # #year-averaged-plots
        # # yr_2011_2020 = np.mean(y_valid[-10*12:, :, :], axis=0)
        # # yr_2011_2020_pred = np.mean(y_valid_pred[-10*12: , :, :], axis=0)
        # # # eval.plot(yr_JAN2014,folder_saving, "model_JAN2014_sla", index = -12)
        # # # eval.plot(yr_JAN2014_pred, folder_saving,"predicted_JAN2014_sla", index = -12)
        # # eval.plot(yr_2011_2020 - yr_2011_2020_pred, folder_saving, "model-predicted_2011_2020_sla", trend=True)
        # #
        # #
        #
        # ## plot persistence
        # y_persistence = y_train[-30*12:,:,:] ##1961-1990
        # persistence_trend = eval.fit_trend(y_persistence, valid_mask, yearly=yearly)
        # eval.plot(persistence_trend, folder_saving, "persistence_trend_1991-2020_same_yaxis", trend=True)
        # model_trend = eval.fit_trend(y_valid, valid_mask, yearly=yearly)
        # # eval.plot(model_trend, folder_saving, "model_trend_1991-2020_same_y_axis", trend=True)
        # eval.plot(model_trend - persistence_trend, folder_saving, "diff_w_persis_trend_1991-2020_same_y_axis", trend=True)


if __name__=='__main__':
    main()
