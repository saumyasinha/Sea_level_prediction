import numpy as np
from math import sqrt
import netCDF4
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
# import cartopy.crs as ccrs


def evaluation_metrics(pred, target, mask):
    diff = target - pred
    diff = diff[mask] ## at this point I've lost the lat/long knowledge and the array is flattened
    print(np.max(diff), np.min(diff))
    l2loss = (diff ** 2).mean()
    rmse = sqrt(l2loss)

    mae = np.abs(diff).mean()

    return rmse, mae


def fit_trend(pred,target,mask):

    lon = pred.shape[0]
    lat = pred.shape[1]

    fitted_coeff = np.full((lon,lat),1e+36)
    for i in range(lon):
        for j in range(lat):
            mask_i_j = mask[:,i,j]
            masked_pred, masked_target = pred[:,i,j][mask_i_j] , target[:,i,j][mask_i_j]
            if len(masked_pred) >0:
                fitted_all_coeffs = poly.polyfit(masked_pred, masked_target,1)
                fitted_coeff[i,j] = fitted_all_coeffs[1]

    return fitted_coeff





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

    # obs_nc = "/Users/saumya/Desktop/Sealevelrise/Data/Forced_Responses/zos/1850-2014/nc_files/historical_MPI-ESM1-2-HR_zos_fr_1850_2014.bin.nc"
    obs_nc="/Users/saumya/Desktop/Sealevelrise/Data/Forced_Responses/zos/2015-2100/nc_files/ssp370_MPI-ESM1-2-HR_zos_fr_2015_2100.bin.nc"
    dataset = netCDF4.Dataset(obs_nc)

    # for var in dataset.variables.values():
    #     print(var)
    #
    if trend==False:
        zos_gt = dataset.variables['zos'][71,:, :] #-12 for jan2014 71 for DEC2020
        print(np.min(zos_gt), np.max(zos_gt), zos_gt.shape)
        zos = np.transpose(xr)
        # zos = xr
        print(np.min(zos), np.max(zos), zos.shape)
        zos = np.ma.masked_where(np.ma.getmask(zos_gt), zos)
        print(np.min(zos), np.max(zos), zos.shape)
        print(type(zos), type(zos_gt))

    # diff = zos_gt - zos

    else:
        zos = np.ma.masked_where(1e+36, xr)

    lats = dataset.variables['lat'][:]
    print(lats.min(), lats.max())
    lons = dataset.variables['lon'][:]
    print(lons.min(), lons.max())

    ax = plt.axes(projection=ccrs.PlateCarree())

    plt.contourf(lons,lats, zos, 60, cmap="jet",
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




