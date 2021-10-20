import numpy as np
import netCDF4
import os, gzip, shutil
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

path_project = "/Users/saumya/Desktop/Sealevelrise/"
path_data = path_project+"Data/"
path_data_obs = path_data + "Observations/"

#
# def plot(obs_nc):
#     dataset = netCDF4.Dataset(obs_nc)
#
#     for var in dataset.variables.values():
#         print(var)
#
#     sla = dataset.variables['sla'][0, :, :]
#     lats = dataset.variables['latitude'][:]
#     print(lats.min(),lats.max())
#     lons = dataset.variables['longitude'][:]
#     print(lons.min(), lons.max())
#     ax = plt.axes(projection=ccrs.PlateCarree())
#
#     plt.contourf(lons, lats, sla, 60,cmap="jet",
#                  transform=ccrs.PlateCarree())
#
#     ax.coastlines()
#     plt.colorbar()
#
#     plt.savefig("model_1993")
# plot("/Users/saumya/Desktop/Sealevelrise/Data/Observations/1993/dt_global_allsat_msla_h_y1993_m01.nc")



def remove_land_values(xr):

    print(np.max(xr), np.min(xr))
    missing_val = -2147483648
    xr[xr == missing_val] = 0
    print(np.max(xr),np.min(xr), np.isnan(xr).sum())
    return xr


def read_nc_files(path_data_obs,year):
    obs_stacked_array = []
    yearfile = path_data_obs + year + "/"
    for monthfile in os.listdir(yearfile):
        if not monthfile.startswith('.'):
            print(monthfile)
            fp = yearfile + monthfile
            nc = netCDF4.Dataset(fp)
            #
            # for var in nc.variables.values():
            #     print(var)

            sla = np.array(nc.variables['sla'][:])  # type(nc.variables))
            print(sla.shape)

            obs_stacked_array.append(sla)

    print(len(obs_stacked_array))
    obs_stacked_array = np.concatenate(obs_stacked_array)
    print(obs_stacked_array.shape)

    return obs_stacked_array


def main():

    total_obs_array = []
    for year in os.listdir(path_data_obs):
        if not year.startswith('.') and not year.endswith('.npy'):
            print(year)
            year_array = read_nc_files(path_data_obs,year)
            print(year_array.shape)
            total_obs_array.append(year_array)

    total_obs_array = np.concatenate(total_obs_array)
    print(total_obs_array.shape)

    xr = remove_land_values(total_obs_array)
    np.save(path_data_obs + 'observations.npy', xr)

if __name__=='__main__':
    main()






