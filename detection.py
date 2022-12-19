'''
Find the LLJs for the whole year
'''
# Import packages
import glob
import datetime
import os
from netCDF4 import Dataset
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import pandas as pd
import utils
import warnings
warnings.filterwarnings('ignore')
from functools import partial

# Read in data

# Directory with wrfouts - 09-12 for 2019, 01-08 for 2020
nwf_data_dir_2019 = "/pl/active/WRFLES_GAD/N_Atl/reruns_Beiter/wrfouts/nwf/2019/"
nwf_data_dir_2020 = "/pl/active/WRFLES_GAD/N_Atl/reruns_Beiter/wrfouts/nwf/2020/"

# Select data from the entire year in domain 2
nwf_2019 = []
for i in range(9, 13):
    nwf_2019 = nwf_2019 + sorted([f for f in glob.glob(nwf_data_dir_2019+f"/{i:02}/wrfout_d02_2019-*")])
nwf_2020 = []
for i in range(1, 9):
    nwf_2020 = nwf_2020 + sorted([f for f in glob.glob(nwf_data_dir_2020+f"/{i:02}/wrfout_d02_2020-*")])
nwf_files_all = sorted(nwf_2019 + nwf_2020)
nwf_files_all = nwf_files_all[::6]  # hourly data

# set up preprocessing to avoid issues with variables that do not match
variables = ['XLAT', 'XLONG', 'XTIME', 'U', 'V', 'PH', 'PHB', 'HGT']
def _preprocess(x):
    return x[variables]
partial_func = partial(_preprocess)

# loop through all of these files to create dataframes for each 6 day period

for period in range(0, 61):  # there are 61 six day period in a leap year
    # select a 144 hour long period
    nwf_files = nwf_files_all[period*144:(period*144)+144]
    
    # try to open the dataset "normally"
    # if there is an error, try to preprocess the data first
    try:
        nwf_ds = xr.open_mfdataset(nwf_files, 
                                   concat_dim = 'Time',
                                   combine = 'nested',
                                   parallel = True,
                                   engine = 'netcdf4',
                                   chunks = {'Time':1})
    except Exception as ex:
        print(f"An error occurred: {ex}")
        print("Trying alternate method")
        nwf_ds = xr.open_mfdataset(nwf_files, 
                                   concat_dim = 'Time',
                                   combine = 'nested',
                                   parallel = True,
                                   engine = 'netcdf4',
                                   chunks = {'Time':1},
                                   preprocess=partial_func)
    finally:
        # Read in latitude, longitude, time, height, and winds
        nwf_lats   = nwf_ds['XLAT'] 
        nwf_lons   = nwf_ds['XLONG']  
        nwf_times  = nwf_ds['XTIME']  
        nwf_U      = nwf_ds['U']      
        nwf_V      = nwf_ds['V']
    
    # find windspeed
    nwf_U = utils.destagger(nwf_ds['U'], 3)
    nwf_U = nwf_U.rename({'west_east_stag': 'west_east'})
    nwf_V = utils.destagger(nwf_ds['V'], 2)
    nwf_V = nwf_V.rename({'south_north_stag': 'south_north'})
    nwf_wspd = np.sqrt((nwf_U**2)+(nwf_V**2))
    
    # Select location
    
    # Select lat/lon values
    lon_vals = nwf_lons.sel(Time=9)
    lat_vals = nwf_lats.sel(Time=9)

    # Find index of lat lon values 
    lon_idx = utils.find_lat_lon_idx(lon_vals, lat_vals, lon=-73.623716, lat=40.711347)[1]
    lat_idx = utils.find_lat_lon_idx(lon_vals, lat_vals, lon=-73.623716, lat=40.711347)[0]

    # Find actual lat/lon values
    lon_val = float(lon_vals.sel(west_east=lon_idx, south_north=lat_idx).values)
    lat_val = float(lat_vals.sel(west_east=lon_idx, south_north=lat_idx).values)

    # Select this location in wspd data
    wspd = nwf_wspd.sel(south_north=lat_idx, west_east=lon_idx)
    U = nwf_U.sel(south_north=lat_idx, west_east=lon_idx)
    V = nwf_V.sel(south_north=lat_idx, west_east=lon_idx)
    
    # find heights

    nwf_PH = utils.destagger(nwf_ds['PH'], 1)
    nwf_PH = nwf_PH.rename({'bottom_top_stag': 'bottom_top'})
    nwf_PH = nwf_PH.sel(south_north=lat_idx, west_east=lon_idx)

    nwf_PHB = utils.destagger(nwf_ds['PHB'], 1)
    nwf_PHB = nwf_PHB.rename({'bottom_top_stag': 'bottom_top'})
    nwf_PHB = nwf_PHB.sel(south_north=lat_idx, west_east=lon_idx)

    nwf_HGT = nwf_ds.HGT.sel(south_north=lat_idx, west_east=lon_idx)
    nwf_z = np.array((nwf_PH+nwf_PHB)/9.81-nwf_HGT)
    
    # Subset data to just 2km and below
    wspd = wspd.sel(bottom_top=slice(0, 31))
    U = U.sel(bottom_top=slice(0, 31))
    V = V.sel(bottom_top=slice(0, 31))

    # LLJ Detection
    
    times = nwf_times.values

    # Create empty lists for each column
    LLJ_classifications = []
    nose_wspds = []
    nose_heights = []
    rotor_shears = []
    rotor_shears_divm = []
    rotor_veers = []
    rotor_veers_divm = []
    sfc_nose_shears = []
    sfc_nose_shears_divm = []
    sfc_nose_veers = []
    sfc_nose_veers_divm = []
    nose_min_shears = []
    nose_min_shears_divm = []
    nose_min_veers = []
    nose_min_veers_divm = []
    sfc_rotortop_shears = []
    sfc_rotortop_shears_divm = []
    sfc_rotortop_veers = []
    sfc_rotortop_veers_divm = []
    wind_dirs_nose = []
    
    # Loop through each time
    for i in range(len(times)):
        # Windspeed and height at this time
        wspd_data = wspd.sel(Time=i)
        heights = nwf_z[i]
        U_now = U.sel(Time=i)
        V_now = V.sel(Time=i)

        # determine if there is an LLJ - go to next time if there is not an LLJ
        LLJ, nose_wspd, nose_height, nose_min_shear, classification, dz, veer = utils.isLLJ(wspd_data, 
                                                                                            heights, 
                                                                                            U_now, 
                                                                                            V_now)                        
        if not LLJ:
            LLJ_classifications.append(np.nan)
            nose_wspds.append(np.nan)
            nose_heights.append(np.nan)
            nose_min_shears.append(np.nan)
            nose_min_shears_divm.append(np.nan)
            nose_min_veers.append(np.nan)
            nose_min_veers_divm.append(np.nan)
            rotor_shears.append(np.nan)
            rotor_shears_divm.append(np.nan)
            rotor_veers.append(np.nan)
            rotor_veers_divm.append(np.nan)
            sfc_nose_shears.append(np.nan)
            sfc_nose_shears_divm.append(np.nan)
            sfc_nose_veers.append(np.nan)
            sfc_nose_veers_divm.append(np.nan)
            sfc_rotortop_shears.append(np.nan)
            sfc_rotortop_shears_divm.append(np.nan)
            sfc_rotortop_veers.append(np.nan)
            sfc_rotortop_veers_divm.append(np.nan)
            wind_dirs_nose.append(np.nan)
            continue

        # if there is an LLJ, make many more calculations...

        # First, add all values found in initial calculation
        LLJ_classifications.append(classification)
        nose_wspds.append(nose_wspd)
        nose_heights.append(nose_height)
        nose_min_shears.append(nose_min_shear)
        nose_min_shears_divm.append(nose_min_shear / dz)
        nose_min_veers.append(veer)
        nose_min_veers_divm.append(veer / dz)

        # Now find shear and veer around rotor region: 30m - 245m
        rotor_shear, rotor_veer, dz = utils.find_shear_veer(30, 245, wspd_data, heights, U_now, V_now)
        rotor_shears.append(rotor_shear)
        rotor_veers.append(rotor_veer)
        rotor_shears_divm.append(rotor_shear / dz) 
        rotor_veers_divm.append(rotor_veer / dz)

        # Find shear and veer for surface to nose height: 0-nose_height
        sfc_nose_shear, sfc_nose_veer, dz = utils.find_shear_veer(0, nose_height, wspd_data, heights, U_now, V_now)
        sfc_nose_shears.append(sfc_nose_shear)
        sfc_nose_shears_divm.append(sfc_nose_shear / dz)
        sfc_nose_veers.append(sfc_nose_veer)
        sfc_nose_veers_divm.append(sfc_nose_veer / dz)

        # Find shear and veer for surface to rotor-top: 0-250
        sfc_rotortop_shear, sfc_rotortop_veer, dz = utils.find_shear_veer(0, 245, wspd_data, heights, U_now, V_now)
        sfc_rotortop_shears.append(sfc_rotortop_shear)
        sfc_rotortop_shears_divm.append(sfc_rotortop_shear / dz)
        sfc_rotortop_veers.append(sfc_rotortop_veer)
        sfc_rotortop_veers_divm.append(sfc_rotortop_veer / dz)

        # Find wind direction around nose:
        nose_idx = list(heights).index(nose_height)
        wind_dir_nose = mpcalc.wind_direction(U_now.sel(bottom_top=nose_idx).values*units('m/s'), 
                                              V_now.sel(bottom_top=nose_idx).values*units('m/s'))
        wind_dirs_nose.append(wind_dir_nose.magnitude)
        
    # Create dataframe
    
    df_dict = {'Time': times,
               'LLJ-classification': LLJ_classifications,
               'Nose windspeed (m/s)': nose_wspds,
               'Nose height (meters)': nose_heights,
               'Rotor region shear (m/s)': rotor_shears,
               'Rotor region shear (1/s)': rotor_shears_divm,
               'Rotor region veer (degrees)': rotor_veers,
               'Rotor region veer (degrees/m)': rotor_veers_divm,
               'Surface to nose shear (m/s)': sfc_nose_shears,
               'Surface to nose shear (1/s)': sfc_nose_shears_divm,
               'Surface to nose veer (degrees)': sfc_nose_veers,
               'Surface to nose veer (degrees/m)': sfc_nose_veers_divm,
               'Nose to min windspeed above nose shear (m/s)': nose_min_shears,
               'Nose to min windspeed above nose shear (1/s)': nose_min_shears_divm,
               'Nose to min windspeed above nose veer (degrees)': nose_min_veers,
               'Nose to min windspeed above nose veer (degrees/m)': nose_min_veers_divm,
               'Surface to top of rotor shear (m/s)': sfc_rotortop_shears,
               'Surface to top of rotor shear (1/s)': sfc_rotortop_shears_divm,
               'Surface to top of rotor veer (degrees)': sfc_rotortop_veers,
               'Surface to top of rotor veer (degrees/m)': sfc_rotortop_veers_divm,
               'Wind direction at nose (degrees)': wind_dirs_nose}
    
    df = pd.DataFrame(df_dict)
    df = df.set_index('Time')
    
    # save the dataframe with the loop number
    df.to_csv(f'NWF_long_island_dfs/detect_{period:02}.csv')
    
    print(f"Finished with loop {period}")
