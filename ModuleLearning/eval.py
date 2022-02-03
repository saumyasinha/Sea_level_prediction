import numpy as np
from math import sqrt
import netCDF4
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm, Normalize
from skimage.measure import block_reduce
from cartopy.util import add_cyclic_point



def evaluation_metrics(pred, target, mask, weight_map):

    # weight_map = np.repeat(weight_map[None, ...], len(pred), axis=0)
    diff = (target - pred)

    weighted_diff2 = (diff ** 2) * weight_map

    # best_rmse_pt = np.unravel_index(np.nanargmin(weighted_diff2), weighted_diff2.shape)
    # worse_rmse_pt = np.unravel_index(np.nanargmax(weighted_diff2), weighted_diff2.shape)
    # print(best_rmse_pt, worse_rmse_pt)

    weighted_diff2 = weighted_diff2[mask] ## at this point I've lost the lat/long knowledge and the array is flattened
    weighted_diff = np.abs(diff) * weight_map
    weighted_diff = weighted_diff[mask]
    weights_masked = weight_map[mask]

    # loss = weighted_diff2.mean()

    l2loss =weighted_diff2.sum() / weights_masked.sum()
    rmse = sqrt(l2loss)

    mae = weighted_diff.sum() / weights_masked.sum()

    # if return_best_and_worst:
    #     return rmse, mae, best_rmse_pt, worse_rmse_pt

    return rmse, mae


def signaltonoise(a, axis=None):
    # a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis)
    # return np.where(sd == 0, 0, m/sd)
    return m/sd


def fit_trend(pred, mask, yearly = False):

    lon = pred.shape[1]
    lat = pred.shape[2]
    missing_val = 1e+36
    x = list(range(2041,2071))
    mid_x = np.mean(x)
    x = np.asarray([i-mid_x for i in x])

    n_years = int(pred.shape[0] / 12)

    if yearly:
        n_years = pred.shape[0]

    print(n_years, len(x))

    y = pred
    if yearly==False:
        y = []
        i=0
        for yr in range(0,n_years):
            print(i)
            annual_pred = np.mean(pred[i:i+12,:,:], axis=0)
            print(annual_pred.shape)
            y.append(annual_pred)
            i=i+12

        y = np.stack(y)
    print(y.shape)

    # plot_global_mean_sea_level(x, y)
    # done = False
    # count=0
    fitted_coeff = np.full((lon,lat),np.nan)
    for i in range(lon):
        for j in range(lat):
            y_i_j = y[:,i,j]
            # print(y_i_j.shape)
            mask_i_j = mask[:,i,j]

            if np.all(mask_i_j==True):
                fitted_all_coeffs = poly.polyfit(x, y_i_j,1)
                # if done == False:
                #     coeff = fitted_all_coeffs[1]
                #     intercept = fitted_all_coeffs[0]
                #     fit_eq = coeff * x + intercept
                #     fig = plt.figure()
                #     ax = fig.subplots()
                #     ax.plot(x, fit_eq, color='r', alpha=0.5, label='Linear fit')
                #     ax.plot(x, y_i_j, color='b', label='time series')  # Original data points
                #     ax.set_title('Linear fit at single point')
                #     ax.legend()
                #     count=count+1
                #     plt.savefig("single_point_test_"+str(count))
                #     if count==5:
                #         done=True
                # print(fitted_all_coeffs)
                fitted_coeff[i,j] = fitted_all_coeffs[1]

    return fitted_coeff

def single_point_test(x_i_j, y_i_j, pred, target, years, count, folder_saving):

    pred_i_j = pred[:,x_i_j,y_i_j]
    print(pred_i_j.shape)
    target_i_j = target[:,x_i_j,y_i_j]

    n_years =len(years)
    pred = []
    i = 0
    for yr in range(0, n_years):
        # print(i)
        annual_pred = np.mean(pred_i_j[i:i + 12])
        pred.append(annual_pred)
        i = i + 12

    pred = np.stack(pred)

    target = []
    i = 0
    for yr in range(0, n_years):
        # print(i)
        annual_pred = np.mean(target_i_j[i:i + 12])
        target.append(annual_pred)
        i = i + 12

    target = np.stack(target)
    print(pred.shape,target.shape)
    # years = list(range(2051,2051))
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(years, pred, color='r', label='prediction')
    ax.plot(years, target, color='b', label='true')  # Original data points
    ax.set_title('prediction at a single point')
    ax.legend()


    plt.savefig(folder_saving+"/single_point_test_wrt_trend_for_prediction_"+str(count))


def plot(xr, folder_saving, save_file, trend =False, index = None):

    # sla_masked = np.ma.masked_where(~mask, xr)
    # sla_masked = np.transpose(sla_masked)
    # print(sla_masked.shape, sla_masked.min(), sla_masked.max())
    # lats = np.load(folder_saving+"/lats.npy",allow_pickle=True)
    # lons = np.load(folder_saving+"/lons.npy",allow_pickle=True)
    #
    # ax = plt.axes(projection=ccrs.PlateCarree())
    #
    # plt.contourf(lons, lats, sla_masked, 60, cmap = "jet",
    #              transform=ccrs.PlateCarree())
    #
    # ax.coastlines()
    # plt.colorbar()
    # plt.savefig(folder_saving+"/"+save_file)
    # plt.close()

    # obs_nc = "/Users/saumya/Desktop/Sealevelrise/Data/Forced_Responses/zos/1850-2014/nc_files/historical_CESM1LE_zos_fr_1850_2014.bin.nc"
    obs_nc="/Users/saumya/Desktop/Sealevelrise/Data/Forced_Responses/zos/2015-2100/nc_files/rcp85_CESM1LE_zos_fr_2015_2100.bin.nc"
    dataset = netCDF4.Dataset(obs_nc)

    # for var in dataset.variables.values():
    #     print(var)
    #
    if trend==False:
        zos_gt = dataset.variables['SSH'][index,:, :]/100
        print(np.min(zos_gt), np.max(zos_gt), zos_gt.shape)
        zos = np.transpose(xr)
        # zos = xr
        print(np.min(zos), np.max(zos), zos.shape)
        zos = np.ma.masked_where(np.ma.getmask(zos_gt), zos)
        print(np.min(zos), np.max(zos), zos.shape)
        print(type(zos), type(zos_gt))

    # diff = zos_gt - zos

    else:
        zos = np.transpose(xr)
        print(np.min(zos), np.max(zos), zos.shape)
        # zos = np.ma.masked_where(zos==1e+36, zos)
        zos = np.ma.masked_where(np.isnan(zos), zos)
        print(np.min(zos), np.max(zos), zos.shape)
        zos = zos*1000


        # print("signal to noise ratio: ", signaltonoise(zos))


    lats =dataset.variables['lat'][:]
    print(lats.shape)
    lats = block_reduce(lats, (2,), np.mean)
    print(lats.min(), lats.max(), lats.shape)
    lons = dataset.variables['lon'][:]
    lons = block_reduce(lons, (2,), np.mean)
    print(lons.min(), lons.max(), lons.shape)

    zos, lons = add_cyclic_point(zos, coord=lons)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=210))  #central_longitude=210
    # norm = TwoSlopeNorm(vmin=zos.min(), vcenter=0, vmax=zos.max())
    v_min=-0.25 #zos.min()
    v_max=0.25#zos.max()
    levels = np.linspace(v_min, v_max, 60)
    # norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
    plt.contourf(lons,lats, zos, cmap="jet",vmin=v_min, vmax=v_max, levels=levels,
                 transform=ccrs.PlateCarree(), extend = "both")  #norm=norm,
    cbar = plt.colorbar()
    # cbar.set_ticks(range(v_min, v_max + 1))

    # least_squares_x = [281,183,279,215,181]
    # least_squares_y = [-51,39,-47,37,39]
    #
    # highest_sqaures_x = [79,83,77,333,335]
    # highest_sqaures_y = [-51,-51,-51,47,47]
    #
    # plt.scatter(least_squares_x, least_squares_y, color='blue', marker='o',
    #      transform=ccrs.PlateCarree())
    # plt.scatter(highest_sqaures_x, highest_sqaures_y, color='red', marker='o',
    #          transform=ccrs.PlateCarree())
    # plt.clim(-0.004, 0.008)
    ax.coastlines()
    plt.savefig(folder_saving+"/"+save_file)
    plt.close()

# [140  19]
# 281.0 -51.0
# [91 64]
# 183.0 39.0
# [139  21]
# 279.0 -47.0
# [107  63]
# 215.0 37.0
# [90 64]
# 181.0 39.0
# [39 19]
# 79.0 -51.0
# [41 19]
# 83.0 -51.0
# [38 19]
# 77.0 -51.0
# [166  68]
# 333.0 47.0
# [167  68]
# 335.0 47.0

def combine_image_patches(y_w_patches):


    y = []
    for row in range(0,y_w_patches.shape[0],16):
        # X_row = X_w_patches[row:row+16]
        y_row = y_w_patches[row:row+16]

        # X_sup_row = []
        y_stiched_row = []
        for i in range(0,16,4):
            # X_sup_col = [X_row[i+j] for j in range(4)]
            # X_sup_col = np.concatenate(X_sup_col,axis=1)

            y_stiched_col = [y_row[i + j] for j in range(4)]
            y_stiched_col = np.concatenate(y_stiched_col, axis=1)
            # print(y_stiched_col.shape)
            # X_sup_row.append(X_sup_col)
            y_stiched_row.append(y_stiched_col)

        # print(np.concatenate(y_stiched_row).shape)
        # X.append(np.concatenate(X_sup_row))
        y.append(np.concatenate(y_stiched_row))

    # X = np.concatenate(X)
    y = np.stack(y)

    print(y.shape)
    return y




