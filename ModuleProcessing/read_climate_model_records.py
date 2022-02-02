import numpy as np
import netCDF4
import os, gzip, shutil

path_local = "/Users/saumya/Desktop/Sealevelrise/"
path_cluster = "/pl/active/machinelearning/ML_for_sea_level/"
path_project = path_local
path_data = path_project+"Data/"
path_data_fr = path_data + "Forced_Responses/"


def gz_extract(directory):
    extension = ".gz"
    os.chdir(directory)
    for item in os.listdir(directory): # loop through items in dir
      if item.endswith(extension): # check for ".gz" extension
          gz_name = os.path.abspath(item) # get full path of files
          file_name = (os.path.basename(gz_name)).rsplit('.',1)[0] #get file name for file within
          with gzip.open(gz_name,"rb") as f_in, open(file_name,"wb") as f_out:
              shutil.copyfileobj(f_in, f_out)
          os.remove(gz_name) # delete zipped file


def read_nc_files(path):
    path_nc = path+"/nc_files/"
    path_npy = path+"/npy_files/"

    for filename in os.listdir(path_nc):
        if filename.endswith(".nc"):
            print(filename)
            fp = path_nc+filename
            nc = netCDF4.Dataset(fp)
            #
            for var in nc.variables.values():
                print(var)

            zos = np.array(nc.variables['heatfull'][:])  #cesm-SSH
            print(zos.shape)
            zos = np.transpose(zos)
            print(zos.shape, np.min(zos), np.max(zos))
            # print(zos[:5, :5, :5])
            np.save(path_npy + filename[:-7] + '.npy', zos)

            os.remove(path_nc + filename)

def read_binary_files(path):
    nlat = 180
    nlon = 360

    # dt = np.dtype((np.float32, (nlat,nlon)))

    for filename in os.listdir(path):
        if filename.endswith(".bin"):
            print(filename)
            with open(path + filename, "rb") as f:
                # xr = np.fromfile(f, dt)
                xr = np.fromfile(f,np.float32)
                print(xr.shape)
                xr = xr.reshape((nlon, nlat, -1))
                print(xr.shape)
                print(xr[:5,:5,:5])
                # xr = remove_land_values(xr)
                np.save(path+filename[:-4]+'.npy',xr)

            os.remove(path + filename)


def get_weights_perpixel(path):

    path_nc = path+"/nc_files/"
    print(path_nc)
    for filename in os.listdir(path_nc):
        if filename.endswith(".nc"):
            fp = path_nc + filename
            nc = netCDF4.Dataset(fp)

            lats = np.array(nc.variables['lat'][:])
            lon = np.array(nc.variables['lon'][:])

            np.save(path_nc+"latitudes.npy",lats)
            np.save(path_nc + "longitudes.npy", lon)
            print(lats.shape, lon.shape)
            for var in nc.variables.values():
                print(var)


            # weights_per_lat = np.cos(lats)
            # # norm_weights = weights_per_lat/weights_per_lat.sum()
            #
            # weight_map = np.full((360,180),0.0)
            # for i in range(180):
            #     weight_map[:,i] = weights_per_lat[i]
            # print(weight_map.shape)
            # print(weight_map[0,0], weight_map[1,0], weights_per_lat[0])
            # # print(np.max(weights), np.min(weights))
            # # norm_weights = weights/weights.sum()
            # # print(np.max(norm_weights), np.min(norm_weights))
            #
            # np.save(path +"/npy_files/weights_" + filename[:-7] + '.npy', weight_map)

            break


def main():

    path_sealevel_folder = path_data_fr + "zos/"
    path_heatcontent_folder = path_data_fr + "heatfull/"

    path_folder =  path_heatcontent_folder #path_sealevel_folder

    historical_path = path_folder + "1850-2014/"
    future_path = path_folder + "2015-2100/"
    #
    # gz_extract(historical_path)
    # gz_extract(future_path)
    #
    # # read_binary_files(historical_path)
    # # read_binary_files(future_path)

    read_nc_files(historical_path)
    read_nc_files(future_path)

    # get_weights_perpixel(historical_path)



if __name__=='__main__':
    main()
