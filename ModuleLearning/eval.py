import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs


def evaluation_metrics(pred, target, mask):
    diff = target - pred
    diff = diff[mask] ## at this point I've lost the lat/long knowledge and the array is flattened
    print(np.max(diff), np.min(diff))
    l2loss = (diff ** 2).mean()
    rmse = sqrt(l2loss)

    mae = np.abs(diff).mean()

    return rmse, mae




def plot(xr, mask, save_file = "model_2021JAN_sla"):


    # fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    #
    # ax.imshow(xr_masked, extent = [-180,180,-90,90],cmap=plt.cm.jet,vmin= xr_masked.min(), vmax= xr_masked.max(),
    #              transform=ccrs.PlateCarree(), origin='lower', interpolation='none')
    #
    # ax.coastlines()
    # ax.colorbar()
    # # cbar.cmap.set_over('green')
    # plt.savefig("true_1993JAN_sla")
    sla_masked = np.ma.masked_where(~mask, xr)
    sla_masked = np.transpose(sla_masked)
    print(sla_masked.shape, sla_masked.min(), sla_masked.max())
    lats = list(range(-89, 91))
    lons = list(range(-180, 180))

    ax = plt.axes(projection=ccrs.PlateCarree())

    plt.contourf(lons, lats, sla_masked, 60, cmap = "jet",
                 transform=ccrs.PlateCarree())

    ax.coastlines()
    plt.colorbar()
    plt.savefig(save_file)
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
            print(y_stiched_col.shape)
            # X_sup_row.append(X_sup_col)
            y_stiched_row.append(y_stiched_col)

        # print(np.concatenate(y_stiched_row).shape)
        # X.append(np.concatenate(X_sup_row))
        y.append(np.concatenate(y_stiched_row))

    # X = np.concatenate(X)
    y = np.stack(y)

    print(y.shape)
    return y




