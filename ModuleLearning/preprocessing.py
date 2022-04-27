import numpy as np
import ModuleLearning.eval as eval
from skimage.measure import block_reduce

def remove_land_values(xr):

    missing_val = 1e+36
    print(np.nanmax(xr), np.nanmin(xr))
    # xr[xr == missing_val] = 0
    xr[np.isnan(xr)] = 0
    print(np.max(xr),np.min(xr))
    return xr


def include_prev_timesteps(X, n_timesteps, include_heat=False):


    X = np.expand_dims(X, axis=2)

    X_with_prev_timesteps = []
    n_months = X.shape[3]
    for i in range(12,n_months): #12 because we always want to start from 1930, and choose what tiemsteps to take from 1929
        prev_list =[]
        for j in range(n_timesteps,0,-1):
            prev = X[:, :, :, i - j]
            if include_heat:
                prev = X[:, :, :, i - j, :]

            prev_list.append(prev)

        curr = X[:,:,:,i]
        if include_heat:
            curr = X[:, :, :, i, :]
        prev_list.append(curr)
        temp = np.concatenate(prev_list, axis=2)
        X_with_prev_timesteps.append(temp)

    # print(len(X_with_prev_timesteps))

    X = np.array(X_with_prev_timesteps)
    # print(X.shape)
    return X


def create_labels(train, test, lead_years, n_prev_months):

    n_months_train = train.shape[2]
    n_months_test = test.shape[2]
    full_array = np.concatenate([train, test[:,:,n_prev_months:]], axis=2) #remove the overlapping prev months from test

    X_train = train[:,:,:]
    y_train = full_array[:,:, lead_years*12:n_months_train+(lead_years*12)]

    X_test = test[:, :, :-lead_years * 12]
    y_test = test[:, :, lead_years * 12:]


    print("X_train, y_train shapes", X_train.shape, y_train.shape)
    print("X_test, y_test shapes", X_test.shape, y_test.shape)


    return  X_train, y_train, X_test, y_test


def year_start_index(year):
    idx = (year - 1850) * 12
    return idx

def create_train_test_split(model, path_1850_to_2014, path_2015_to_2100, train_start_year, train_end_year, test_start_year, test_end_year, lead_years, downscaling=False, heat=False, n_prev_months=12):

    '''
    if train_start_year = 1900, train_end_year = 1990,
    test_start_year = 1991, test_end_year = 2020
    and  lead_years = 30, n_prev_months = 12
    then this function returns train: 1899 - 1990, test: 1990 - 2050
    '''
    historical_filename = path_1850_to_2014 + "historical_"+model+"_zos_fr_1850_2014.npy"
    if heat:
        historical_filename = path_1850_to_2014 + "historical_" + model + "_heatfull_fr_1850_2014.npy"
    historical_array = np.load(historical_filename)
    print(historical_array.shape)


    future_filename = path_2015_to_2100 + "ssp370_" + model + "_zos_fr_2015_2100.npy" #ssp370/rcp85
    if model == 'CESM1LE':
        future_filename = path_2015_to_2100 + "rcp85_" + model + "_zos_fr_2015_2100.npy"

    if heat:
        future_filename = path_2015_to_2100 + "ssp370_" + model + "_heatfull_fr_2015_2100.npy"
    future_array = np.load(future_filename)
    print(future_array.shape)

    full_array = np.concatenate([historical_array,future_array], axis=2)

    #
    train_start_index = year_start_index(train_start_year)
    train_end_index = year_start_index(train_end_year)+12
    train = full_array[:, :,train_start_index-n_prev_months: train_end_index]
    print(train_start_index,train_end_index,train.shape)

    test_start_index = year_start_index(test_start_year)
    test_end_index = year_start_index(test_end_year+lead_years)+12
    test = full_array[:, :, test_start_index-n_prev_months: test_end_index]
    print(test_start_index, test_end_index, test.shape)

    ##masking land values with np.nan
    train[train == 1e+36] = np.nan
    test[test == 1e+36] = np.nan

    if downscaling:
        train = downscale_input(train)
        test = downscale_input(test)

    print(np.nanmin(train), np.nanmax(train), np.nanmin(test), np.nanmax(test), np.isnan(train).sum(), np.isnan(test).sum())
    return train, test

def create_train_test_and_labels_on_trends(train, test, folder_saving,  lead_years=30, trend_over_years=30, n_prev_months=12):

    # print("inside trends function")
    # trend_over_months = trend_over_years*12
    # n_months_train = train.shape[2]
    # n_months_test = test.shape[2]
    # full_array = np.concatenate([train, test[:, :, n_prev_months:]], axis=2)
    # print(full_array.shape, n_months_train, n_months_test)
    #
    # # full_array = full_array
    # full_trend_array = []
    # for month_idx in range(full_array.shape[2]-trend_over_months+1):
    #     data_to_trend_on = full_array[:,:,month_idx:month_idx+trend_over_months]
    #     data_to_trend_on = np.transpose(data_to_trend_on, (2, 0, 1))
    #     print(data_to_trend_on.shape)
    #     mask = ~np.isnan(data_to_trend_on)
    #     trended_map = eval.fit_trend(data_to_trend_on, mask, yearly=False)
    #     print(trended_map.shape,mask.shape)
    #     full_trend_array.append(trended_map[:,:,np.newaxis])
    #
    # full_trend_array = np.concatenate(full_trend_array, axis=2)
    # print(full_trend_array.shape)
    # np.save(folder_saving+"/full_trended_array.npy",full_trend_array)
    # print(np.nanmin(full_trend_array), np.nanmax(full_trend_array), np.isnan(full_trend_array).sum())
    full_trend_array = np.load(folder_saving+"/full_trended_array.npy")
    ## we remove the last point, to keep the data from Jan 1929 to Dec 2070
    full_trend_array = full_trend_array[:,:,:-1]

    print("full trend array shape", full_trend_array.shape)

    ## keeping the last 10 years as test, train: 102 yrs and test is 10 yrs from trend full data of 112 years of labelled points
    n_months_train = 102*12
    n_months_test = 10*12
    X_train = full_trend_array[:, :, :n_months_train]
    y_train = full_trend_array[:, :, lead_years * 12:n_months_train + (lead_years * 12)]

    X_test = full_trend_array[:, :, n_months_train-12:n_months_train + n_months_test] ## doing -12 here to include one year before in the test data
    y_test = full_trend_array[:, :, n_months_train-12+lead_years * 12:n_months_train + n_months_test+lead_years * 12]

    print("X_train, y_train shapes", X_train.shape, y_train.shape)
    print("X_test, y_test shapes", X_test.shape, y_test.shape)

    print(np.nanmin(X_train), np.nanmax(X_train), np.nanmin(X_test), np.nanmax(X_test), np.isnan(X_train).sum(),
          np.isnan(X_test).sum())

    print(np.nanmin(y_train), np.nanmax(y_train), np.nanmin(y_test), np.nanmax(y_test), np.isnan(y_train).sum(),
          np.isnan(y_test).sum())

    return X_train, y_train, X_test, y_test

def create_train_test_and_labels_on_ssh_averages(train, test, folder_saving,  lead_years=30, average_over_years=30, n_prev_months=12):

    average_over_months = average_over_years*12
    full_array = np.concatenate([train, test[:, :, n_prev_months:]], axis=2)
    print(full_array.shape)

    full_averaged_array = []
    for month_idx in range(full_array.shape[2]-average_over_months+1):
        data_to_average_on = full_array[:,:,month_idx:month_idx+average_over_months]
        averaged_map = np.mean(data_to_average_on, axis=2)
        full_averaged_array.append(averaged_map[:,:,np.newaxis])

    full_averaged_array = np.concatenate(full_averaged_array, axis=2)
    print(full_averaged_array.shape)
    # np.save(folder_saving+"/full_averaged_array.npy",full_averaged_array)
    print(np.nanmin(full_averaged_array), np.nanmax(full_averaged_array), np.isnan(full_averaged_array).sum())


    # full_averaged_array = np.load(folder_saving+"/full_averaged_array.npy")
    # we remove the last point, to keep the data from Jan 1929 to Dec 2070
    full_averaged_array = full_averaged_array[:,:,:-1]
    full_averaged_array = full_averaged_array/100 #to convert it to meters
    print("full averaged array shape", full_averaged_array.shape)

    # keeping the last 10 years as test, train: 102 yrs and test is 10 yrs from trend full data of 112 years of labelled points
    n_months_train = 102*12
    n_months_test = 10*12
    X_train = full_averaged_array[:, :, :n_months_train]
    y_train = full_averaged_array[:, :, lead_years * 12:n_months_train + (lead_years * 12)]

    X_test = full_averaged_array[:, :, n_months_train-12:n_months_train + n_months_test] ## doing -12 here to include one year before in the test data
    y_test = full_averaged_array[:, :, n_months_train-12+lead_years * 12:n_months_train + n_months_test+lead_years * 12]

    print("X_train, y_train shapes", X_train.shape, y_train.shape)
    print("X_test, y_test shapes", X_test.shape, y_test.shape)

    print(np.nanmin(X_train), np.nanmax(X_train), np.nanmin(X_test), np.nanmax(X_test), np.isnan(X_train).sum(),
          np.isnan(X_test).sum())

    print(np.nanmin(y_train), np.nanmax(y_train), np.nanmin(y_test), np.nanmax(y_test), np.isnan(y_train).sum(),
          np.isnan(y_test).sum())

    return X_train, y_train, X_test, y_test


def convert_month_to_years(X):
    n_months = X.shape[2]
    n_years = int(n_months / 12)

    X_yrs = []
    i = 0
    for yr in range(0, n_years):
        X_sub = X[:,:,i:i + 12]
        annual_X = np.mean(X_sub, axis=2)
        # annual_X[annual_X>1e+36]=1e+36
        # print(annual_X.shape)
        X_yrs.append(annual_X)
        i = i + 12

    X_yrs = np.stack(X_yrs, axis=2)
    print(X_yrs.shape)

    return X_yrs

def get_image_patches(X,y):

    if X is not None:
        X_with_patches = []
    y_with_patches = []
    for row in range(y.shape[0]):
        if X is not None:
            X_row = X[row]
            subs_X = []

        y_row = y[row]
        subs_Y = []

        i=0
        while i<360:
            j=0
            while j<180:
                if X is not None:
                    subX = X_row[i:i+90,j:j+45,:]
                    subs_X.append(subX)


                subY = y_row[i:i+90,j:j+45]
                subs_Y.append(subY)

                j = j+45

            i = i + 90

        if X is not None:
            X_with_patches.append(subs_X)

        y_with_patches.append(subs_Y)
    # print(X_with_patches[0][0].shape)
    if X is not None:
        X_with_patches = np.concatenate(X_with_patches)
    y_with_patches = np.concatenate(y_with_patches)

    if X is not None:
        print(X_with_patches.shape)
        print(y_with_patches.shape)
        return X_with_patches,y_with_patches
    else:
        print(y_with_patches.shape)
        return y_with_patches


def downscale_input(x):

    x_down = block_reduce(x, (2,2,1), np.mean)#(2,2,1)
    print(x_down.shape)

    return x_down


def normalize_from_train(X_train, X_test,y_train, y_test, split_index):

    X_pure_train = X_train[:,:,:-split_index]
    n_months = X_pure_train.shape[2]

    n_months_test = X_test.shape[2]

    print(np.nanmin(y_test),np.nanmin(y_train), np.nanmin(X_test), np.nanmin(X_train))

    for month in range(12):
        indices_month = [i for i in range(n_months) if i%12==month]
        X_sub = X_pure_train[:, :, indices_month]
        # print(X_sub.shape)
        avg_for_month = np.mean(X_sub, axis=2)
        # std_for_month = np.std(X_sub, axis=2)
        # print(avg_for_month.shape, avg_for_month[:,:,np.newaxis].shape)
        X_train[:,:,indices_month] = X_train[:,:,indices_month] - avg_for_month[:,:,np.newaxis]
        # y_train[:, :, indices_month] = y_train[:, :, indices_month] - avg_for_month[:, :, np.newaxis]


        indices_month_test = [i for i in range(n_months_test) if i%12==month]
        X_test[:, :, indices_month_test] = X_test[:, :, indices_month_test] - avg_for_month[:,:,np.newaxis]
        # y_test[:, :, indices_month_test] = y_test[:, :, indices_month_test] - avg_for_month[:, :, np.newaxis]


    return X_train, X_test, y_train, y_test

# def get_climatology_data(y_train, folder_path):
#     cliamtology_data = y_train[10*12 : (22*12)+12, :, :]
#     n_months = cliamtology_data.shape[0]
#     print(cliamtology_data.shape)
#
#     final_climatology_avg_data = []
#
#     for month in range(12):
#         indices_month = [i for i in range(n_months) if i % 12 == month]
#         X_sub = cliamtology_data[indices_month, :, :]
#         print(X_sub.shape)
#         avg_for_month = np.mean(X_sub, axis=0)
#         print(avg_for_month.shape)
#         final_climatology_avg_data.append(avg_for_month)
#
#
#
#     final_climatology_avg_data = np.array(final_climatology_avg_data)
#     print(final_climatology_avg_data.shape)
#
#     np.save(folder_path+"/climatology_cesm.npy", final_climatology_avg_data)
