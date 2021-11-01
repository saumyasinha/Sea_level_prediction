import numpy as np
from math import sqrt
import netCDF4
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
# import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm, Normalize


def evaluation_metrics(pred, target, mask):
    diff = target - pred
    diff = diff[mask] ## at this point I've lost the lat/long knowledge and the array is flattened
    print(np.max(diff), np.min(diff))
    l2loss = (diff ** 2).mean()
    rmse = sqrt(l2loss)

    mae = np.abs(diff).mean()

    return rmse, mae


def fit_trend(pred, mask, yearly = False):

    lon = pred.shape[1]
    lat = pred.shape[2]
    missing_val = 1e+36
    x = list(range(2001,2021))
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

    fitted_coeff = np.full((lon,lat),missing_val)
    for i in range(lon):
        for j in range(lat):
            y_i_j = y[:,i,j]
            # print(y_i_j.shape)
            mask_i_j = mask[:,i,j]

            if np.all(mask_i_j==True):
                fitted_all_coeffs = poly.polyfit(x, y_i_j,1)
                # print(fitted_all_coeffs)
                fitted_coeff[i,j] = fitted_all_coeffs[1]

    return fitted_coeff

#
# def plot_global_mean_sea_level(x,y,missing_val = 1e+36):
#     y[y==missing_val] = 0
#     global_mean = np.mean(y,axis=(1,2))
#
#     fig, ax = plt.subplots(1)
#     ax.plot(x, global_mean)
#     ax.set_xticks(x)
#     ax.set_yticks(global_mean)
#     plt.savefig("global_mean_trend_2001-2020")
#     plt.close()



def plot(xr, folder_saving, save_file, trend =False):

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

    obs_nc = "/Users/saumya/Desktop/Sealevelrise/Data/Forced_Responses/zos/1850-2014/nc_files/historical_CESM1LE_zos_fr_1850_2014.bin.nc"
    # obs_nc="/Users/saumya/Desktop/Sealevelrise/Data/Forced_Responses/zos/1850-2014/nc_files/historical_MPI-ESM1-2-HR_zos_fr_1850_2014.bin.nc"
    dataset = netCDF4.Dataset(obs_nc)

    # for var in dataset.variables.values():
    #     print(var)
    #
    if trend==False:
        zos_gt = np.mean(dataset.variables['SSH'][-12,:, :]/100, axis=0) #-12 for jan2014 71 for DEC2020
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
        zos = np.ma.masked_where(zos==1e+36, zos)
        print(np.min(zos), np.max(zos), zos.shape)

    lats = dataset.variables['lat'][:]
    print(lats.min(), lats.max())
    lons = dataset.variables['lon'][:]
    print(lons.min(), lons.max())

    zos = zos*1000
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    norm = TwoSlopeNorm(vmin=zos.min(), vcenter=0, vmax=zos.max())
    plt.contourf(lons,lats, zos, 60, norm = norm,  cmap="jet",
                 transform=ccrs.PlateCarree())

    ax.coastlines()
    plt.colorbar()
    plt.savefig(folder_saving+"/"+save_file)
    plt.close()



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




