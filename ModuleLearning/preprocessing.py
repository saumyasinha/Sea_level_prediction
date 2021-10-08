import numpy as np


def remove_land_values(xr):

    missing_val = 1e+36
    print(np.max(xr), np.min(xr))
    xr[xr == missing_val] = 0
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


def create_labels(train, test, lead_years):

    n_months_train = train.shape[2]
    n_months_test = test.shape[2]
    full_array = np.concatenate([train, test], axis=2)

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
    if train_start_year = 1990, train_end_year = 1990,
    test_start_year = 1991, test_end_year = 2020
    and  lead_years = 30, n_prev_months = 12
    then this function returns train: 1989 - 1990, test: 1990 - 2050
    '''
    historical_filename = path_1850_to_2014 + "historical_"+model+"_zos_fr_1850_2014.npy"
    historical_array = np.load(historical_filename)
    print(historical_array.shape)

    future_filename = path_2015_to_2100 + "ssp370_" + model + "_zos_fr_2015_2100.npy"
    future_array = np.load(future_filename)
    print(future_array.shape)

    full_array = np.concatenate([historical_array,future_array], axis=2)
    # print(full_array.shape)

    #
    train_start_index = year_start_index(train_start_year)
    train_end_index = year_start_index(train_end_year)+12
    train = full_array[:, :,train_start_index-n_prev_months: train_end_index]
    print(train_start_index,train_end_index,train.shape)

    test_start_index = year_start_index(test_start_year)
    test_end_index = year_start_index(test_end_year+lead_years)+12
    test = full_array[:, :, test_start_index-n_prev_months: test_end_index]
    print(test_start_index, test_end_index, test.shape)

    print(np.min(train), np.max(train), np.min(test), np.max(test), np.isnan(train).sum(), np.isnan(test).sum())
    return train, test





#
# 'MPI-ESM1-2-HR'
# -2.0821238 1.6953052 -2.0989676 1.7183158
#
# 'MPI-ESM1-2-LR'
# -1.960577 1.7667885 -1.9656749 1.7706753
#
# 'ACCESS-ESM1-5'
# -6.533461 0.0 -6.3966355 0.0
#
#
# X_train, y_train shapes (360, 180, 336) (360, 180, 336)
# X_test, y_test shapes (360, 180, 600) (360, 180, 600)


