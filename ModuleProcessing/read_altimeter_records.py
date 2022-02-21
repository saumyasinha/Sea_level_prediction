import numpy as np
import netCDF4
import os, gzip, shutil
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from skimage.measure import block_reduce

path_project = "/Users/saumya/Desktop/Sealevelrise/"
path_data = path_project+"Data/"
path_data_obs = path_data + "Observations/"
path_models = path_project+"ML_Models/"

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



# def remove_land_values(xr):
#
#     print(np.max(xr), np.min(xr))
#     missing_val = -2147483648
#     xr[xr == missing_val] = 0
#     print(np.max(xr),np.min(xr), np.isnan(xr).sum())
#     return xr


def read_nc_files(path_data_obs,year, add_climatology = False, climatology_input = ""):
    obs_stacked_array = []
    yearfile = path_data_obs + str(year) + "/"
    monthlist = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    if year == 2020:
        monthlist = ["01", "02", "03", "04", "05"]
    month_file_list = ["dt_global_allsat_msla_h_y"+str(year)+"_m"+month+".nc" for month in monthlist]


    for month in monthlist:
        monthfile = "dt_global_allsat_msla_h_y"+str(year)+"_m"+month+".nc"
        # if not monthfile.startswith('.'):
        fp = yearfile + monthfile
        nc = netCDF4.Dataset(fp)
        #
        # for var in nc.variables.values():
        #     print(var)

        sla = np.array(nc.variables['sla'][:])  # type(nc.variables))
        print(sla.shape)
        print(np.min(sla), np.max(sla))
        if add_climatology:
            sla[sla == -2147483648] = np.nan
            print(np.nanmin(sla), np.nanmax(sla))
            sla = block_reduce(sla, (1,8,8), np.mean)
            sla =  np.transpose(sla, (0,2,1))
            print(sla.shape, climatology_input[int(month)-1, :, :].shape)
            climatology_for_the_month = climatology_input[int(month)-1, :, :]
            climatology_for_the_month = climatology_for_the_month[np.newaxis, :, :]
            print(np.nanmin(climatology_for_the_month), np.nanmax(climatology_for_the_month))
            sla = climatology_for_the_month + sla
            print(np.nanmin(sla), np.nanmax(sla))


        obs_stacked_array.append(sla)

    print(len(obs_stacked_array))
    obs_stacked_array = np.concatenate(obs_stacked_array)
    print(obs_stacked_array.shape)

    return obs_stacked_array


def main():

    total_obs_array = []
    years = list(range(1993,2021))

    add_climatology = True
    climatology_input=""
    if add_climatology:
        climatology_input = np.load(path_models+'CESM1LE/'+"climatology_cesm.npy")

    for year in years:
        # if not year.startswith('.') and not year.endswith('.npy'):
        print(year)
        year_array = read_nc_files(path_data_obs,year, add_climatology = add_climatology, climatology_input = climatology_input)
        print(year_array.shape)
        total_obs_array.append(year_array)

    total_obs_array = np.concatenate(total_obs_array)
    print(total_obs_array.shape)

    if add_climatology:
        np.save(path_data_obs + 'observations_with_climatology.npy', total_obs_array)
    else:
        np.save(path_data_obs + 'observations.npy', total_obs_array)

if __name__=='__main__':
    main()






