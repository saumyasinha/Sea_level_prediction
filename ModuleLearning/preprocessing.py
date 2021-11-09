import numpy as np


def remove_land_values(xr):

    missing_val = 1e+36
    print(np.nanmax(xr), np.nanmin(xr))
    # xr[xr == missing_val] = 0
    xr[np.isnan(xr)] = 0
    print(np.max(xr),np.min(xr))
    return xr


def include_prev_timesteps(X, n_timesteps):

    X = np.expand_dims(X, axis=2)
    X_with_prev_timesteps = []
    n_months = X.shape[3]
    for i in range(n_timesteps,n_months):
        prev_list =[]
        for j in range(n_timesteps,0,-1):
            prev = X[:,:,:,i-j]
            prev_list.append(prev)

        curr = X[:,:,:,i]
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

def create_train_test_split(model, path_1850_to_2014, path_2015_to_2100, train_start_year, train_end_year, test_start_year, test_end_year, n_prev_months, lead_years):

    '''
    if train_start_year = 1900, train_end_year = 1990,
    test_start_year = 1991, test_end_year = 2020
    and  lead_years = 30, n_prev_months = 12
    then this function returns train: 1899 - 1990, test: 1990 - 2050
    '''
    historical_filename = path_1850_to_2014 + "historical_"+model+"_zos_fr_1850_2014.npy"
    historical_array = np.load(historical_filename)
    print(historical_array.shape)

    future_filename = path_2015_to_2100 + "rcp85_" + model + "_zos_fr_2015_2100.npy" #ssp370/rcp85
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

    print(np.nanmin(train), np.nanmax(train), np.nanmin(test), np.nanmax(test), np.isnan(train).sum(), np.isnan(test).sum())
    return train, test

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

    X_with_patches = []
    y_with_patches = []
    for row in range(X.shape[0]):
        X_row = X[row]
        y_row = y[row]
        subs_X = []
        subs_Y = []

        i=0
        while i<360:
            j=0
            while j<180:
                subX = X_row[i:i+90,j:j+45,:]
                subY = y_row[i:i+90,j:j+45]

                subs_X.append(subX)
                subs_Y.append(subY)

                j = j+45

            i = i + 90

        X_with_patches.append(subs_X)
        y_with_patches.append(subs_Y)
    # print(X_with_patches[0][0].shape)
    X_with_patches = np.concatenate(X_with_patches)
    y_with_patches = np.concatenate(y_with_patches)

    print(X_with_patches.shape, y_with_patches.shape)
    return X_with_patches,y_with_patches



def normalize_from_train(X_train, X_test):

    n_months = X_train.shape[2]

    for month in range(12):
        indices_month = [i for i in range(n_months) if i%12==month]
        X_sub = X_train[:, :, indices_month]
        print(X_sub.shape)
        avg_for_month = np.mean(X_sub, axis=2)
        std_for_month = np.std(X_sub, axis=2)
        print(avg_for_month.shape, avg_for_month[:,:,np.newaxis].shape)
        X_train[:,:,indices_month] = (X_train[:,:,indices_month] - avg_for_month[:,:,np.newaxis])/std_for_month[:,:,np.newaxis]

        n_months_test = X_test.shape[2]
        indices_month_test = [i for i in range(n_months_test) if i%12==month]
        print(X_test[:, :, indices_month_test].shape)
        X_test[:, :, indices_month_test] = (X_test[:, :, indices_month_test] - avg_for_month[:,:,np.newaxis])/std_for_month[:,:,np.newaxis]


    return X_train,X_test

