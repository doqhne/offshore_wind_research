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
import numpy as np
import argparse
from functools import partial
import time

starttime = time.time()

parser = argparse.ArgumentParser(
                description='Find LLJs in WRF data',
                prog='detection2.py')

parser.add_argument('--sim',
                    type=str,
                    help='Options are [nwf, vw_100_tke, la_100_tke, ca_100_tke]',
                    required=True)
parser.add_argument('--lat',
                    type=float,
                    help='Latitude in degrees north',
                    required=True)
parser.add_argument('--lon',
                    type=float,
                    help='Longitude in degrees east',
                    required=False)
parser.add_argument('--output_file',
                    type=str,
                    help='name of file to save',
                    required=True)

args = parser.parse_args()

# Read in data

# Directory with wrfouts - 09-12 for 2019, 01-08 for 2020
nwf_data_dir_2019 = f"/pl/active/JKL_REAL/N_Atl/reruns_Beiter/wrfouts/{args.sim}/2019/"
nwf_data_dir_2020 = f"/pl/active/JKL_REAL/N_Atl/reruns_Beiter/wrfouts/{args.sim}/2020/"

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
variables = ['XLAT', 'XLONG', 'XTIME', 'U', 'V', 'PH', 'PHB', 'HGT', 'HFX', 'UST', 'T', 'T2', 'QVAPOR', 'PSFC']
def _preprocess(x):
    return x[variables]
partial_func = partial(_preprocess)

# now process all of the files

full_df = pd.DataFrame()

print('File list created', time.time() - starttime)

for period in range(0, 61):  # there are 61 six day period in a leap year
    # select a 144 hour long period
    nwf_files = nwf_files_all[period*144:(period*144)+144] # change to 144
    
    if period != 10:
        ds = xr.open_mfdataset(nwf_files, 
                               concat_dim = 'Time',
                               combine = 'nested',
                               parallel = True,
                               engine = 'netcdf4',
                               chunks = {'Time':1})
    else:
        ds = xr.open_mfdataset(nwf_files, 
                               concat_dim = 'Time',
                               combine = 'nested',
                               parallel = True,
                               engine = 'netcdf4',
                               chunks = {'Time':1},
                               preprocess=partial_func)

    # Read in latitude, longitude, time, height, and winds
    lats   = ds['XLAT'] 
    lons   = ds['XLONG']  
    times  = ds['XTIME'].values
    U      = ds['U']      
    V      = ds['V']
    # variables for Obukhov length and BVF
    ust = ds.UST # friction velocity
    hfx = ds.HFX # vertical turbulent heat flux
    T2 = ds.T2 # 2m temp
    T = ds.T + 300 # potential temp
    r = ds.QVAPOR # mixing ratio
    pres = ds.PSFC / 100 # surface pressure, in mbar

    # find windspeed
    U = utils.destagger(ds.U, 3)
    U = U.rename({'west_east_stag': 'west_east'})
    V = utils.destagger(ds.V, 2)
    V = V.rename({'south_north_stag': 'south_north'})
    wspd = np.sqrt((U**2)+(V**2))
    
    # select location
    
    # Select lat/lon values
    lon_vals = lons.sel(Time=0)
    lat_vals = lats.sel(Time=0)

    # Find index of lat lon values  40.95N, -70.59E
    lon_idx = utils.find_lat_lon_idx(lon_vals, lat_vals, lon=args.lon, lat=args.lat)[1]
    lat_idx = utils.find_lat_lon_idx(lon_vals, lat_vals, lon=args.lon, lat=args.lat)[0]

    # Find actual lat/lon values
    lon_val = float(lon_vals.sel(west_east=lon_idx, south_north=lat_idx).values)
    lat_val = float(lat_vals.sel(west_east=lon_idx, south_north=lat_idx).values)

    # Select this location in data
    wspd = wspd.sel(south_north=lat_idx, west_east=lon_idx)
    U = U.sel(south_north=lat_idx, west_east=lon_idx)
    V = V.sel(south_north=lat_idx, west_east=lon_idx)
    ust = ust.sel(south_north=lat_idx, west_east=lon_idx)
    hfx = hfx.sel(south_north=lat_idx, west_east=lon_idx)
    T = T.sel(south_north=lat_idx, west_east=lon_idx)
    T2 = T2.sel(south_north=lat_idx, west_east=lon_idx)
    pres = pres.sel(south_north=lat_idx, west_east=lon_idx)
    r = r.sel(south_north=lat_idx, west_east=lon_idx)
    
    # find heights

    PH = utils.destagger(ds['PH'], 1)
    PH = PH.rename({'bottom_top_stag': 'bottom_top'})
    PH = PH.sel(south_north=lat_idx, west_east=lon_idx)

    PHB = utils.destagger(ds['PHB'], 1)
    PHB = PHB.rename({'bottom_top_stag': 'bottom_top'})
    PHB = PHB.sel(south_north=lat_idx, west_east=lon_idx)

    HGT = ds.HGT.sel(south_north=lat_idx, west_east=lon_idx)
    z = np.array((PH+PHB)/9.81-HGT)
    del PH, PHB, HGT, lats, lons
    
    # index where heights are closest to 2000m
    # find the index of z at 2000m
    hgts_2k = []
    hgts_1k = []
    for i in range(144):
        hgts_2k.append(len(z[i][z[i]<2200])-1)
        hgts_1k.append(len(z[i][z[i]<1200])-1)
    hgt_2k_idx = np.max(hgts_2k)
    hgt_1k_idx = np.max(hgts_1k)

    # Subset data to just ~2km and below
    wspd = wspd.sel(bottom_top=slice(0, hgt_2k_idx))
    U = U.sel(bottom_top=slice(0, hgt_2k_idx))
    V = V.sel(bottom_top=slice(0, hgt_2k_idx))
    T = T.sel(bottom_top=slice(0, hgt_2k_idx))
    r = r.sel(bottom_top=slice(0, hgt_2k_idx))
    z = z[:, :hgt_2k_idx]
    
    # calculate wind direction
    wdirs = mpcalc.wind_direction(U.values*units('m/s'), V.values*units('m/s')).magnitude

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
    obukhov_lengths = []
    bvfs = []

    # calculate Obukhov length
    rmol = utils.obukhov(ust.values*units('m/s'), 
                         T2.values*units('K'),
                         hfx.values*units('W/m^2'),
                         r.sel(bottom_top=0).values*units('kg/kg'),
                         pres.values*units('mbar')).magnitude
    
    for i in range(len(times)):
        # select data for this timestep
        wspd_i = wspd.sel(Time=i).values
        heights_i = z[i]
        U_i = U.sel(Time=i).values
        V_i = V.sel(Time=i).values
        wdir_i = wdirs[i]

        # find maximum windspeed and index
        ws_max = wspd_i[: hgt_1k_idx+1].max()
        ws_max_idx = np.argmax(wspd_i[: hgt_1k_idx+1])
        # find nose height
        nose_hgt = heights_i[ws_max_idx]
        # find shear above nose
        above_nose_shear = np.ptp(wspd_i[ws_max_idx:])

        # determine if there is an LLJ - if not, add nan values to all columns
        if not utils.is_LLJ(ws_max, nose_hgt, above_nose_shear):
            LLJ_classifications.append(np.nan)
            nose_wspds.append(np.nan)
            nose_heights.append(np.nan)
            wind_dirs_nose.append(np.nan)
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
            obukhov_lengths.append(np.nan)
            bvfs.append(np.nan)
        else:
            # add classification, nose ws, and height to columns
            LLJ_classifications.append(utils.find_classification(ws_max, above_nose_shear))
            nose_wspds.append(ws_max)
            nose_heights.append(nose_hgt)

            # calculate nose wind direction
            wind_dirs_nose.append(wdir_i[ws_max_idx])

            # add above nose shear and veer to columns
            l = ws_max_idx
            u = np.argmin(wspd_i[ws_max_idx:])
            dz = utils.find_dz(heights_i, l, u) + ws_max_idx
            nose_min_shears.append(above_nose_shear)
            nose_min_shears_divm.append(above_nose_shear / dz)
            nose_min_veers.append(utils.find_veer(wdir_i, u, l))
            nose_min_veers_divm.append(utils.find_veer(wdir_i, u, l) / dz)

            # add rotor region shear and veer to columns
            l = (np.abs(heights_i - 30)).argmin()
            u = (np.abs(heights_i - 250)).argmin()
            dz = utils.find_dz(heights_i, l, u)
            rotor_shears.append(wspd_i[u] - wspd_i[l])
            rotor_shears_divm.append(above_nose_shear / dz)
            rotor_veers.append(utils.find_veer(wdir_i, u, l))
            rotor_veers_divm.append(utils.find_veer(wdir_i, u, l) / dz)

            # add surface to nose shear and veer to columns
            l = 0
            u = ws_max_idx
            dz = utils.find_dz(heights_i, l, u)
            sfc_nose_shears.append(wspd_i[u] - wspd_i[l])
            sfc_nose_shears_divm.append(above_nose_shear / dz)
            sfc_nose_veers.append(utils.find_veer(wdir_i, u, l))
            sfc_nose_veers_divm.append(utils.find_veer(wdir_i, u, l) / dz)

            # calcualte BVF - surface to nose
            bvfs.append(utils.BVF(T[i].values*units('K'), dz*units('m'), ws_max_idx))

            # add sfc to rotor top shear and veer to columns
            l = 0
            u = (np.abs(heights_i - 250)).argmin()
            dz = utils.find_dz(heights_i, l, u)
            sfc_rotortop_shears.append(wspd_i[u] - wspd_i[l])
            sfc_rotortop_shears_divm.append(above_nose_shear / dz)
            sfc_rotortop_veers.append(utils.find_veer(wdir_i, u, l))
            sfc_rotortop_veers_divm.append(utils.find_veer(wdir_i, u, l) / dz)

            # calculate obukhov length
            obukhov_lengths.append(float(rmol[i]))

    # Create dictionary to make dataframe
    df_dict = {'Time': times,
               'LLJ-classification': LLJ_classifications,
               'Nose windspeed (m/s)': nose_wspds,
               'Nose height (meters)': nose_heights,
               'Wind direction at nose (degrees)': wind_dirs_nose,
               'Above nose shear (m/s)': nose_min_shears,
               'Above nose shear (1/s)': nose_min_shears_divm,
               'Above nose veer (degrees)': nose_min_veers,
               'Above nose veer (degrees/m)': nose_min_veers_divm,
               'Rotor region shear (m/s)': rotor_shears,
               'Rotor region shear (1/s)': rotor_shears_divm,
               'Rotor region veer (degrees)': rotor_veers,
               'Rotor region veer (degrees/m)': rotor_veers_divm,
               'Surface to nose shear (m/s)': sfc_nose_shears,
               'Surface to nose shear (1/s)': sfc_nose_shears_divm,
               'Surface to nose veer (degrees)': sfc_nose_veers,
               'Surface to nose veer (degrees/m)': sfc_nose_veers_divm,
               'Surface to top of rotor shear (m/s)': sfc_rotortop_shears,
               'Surface to top of rotor shear (1/s)': sfc_rotortop_shears_divm,
               'Surface to top of rotor veer (degrees)': sfc_rotortop_veers,
               'Surface to top of rotor veer (degrees/m)': sfc_rotortop_veers_divm,
               'Obukhov Length (m)': obukhov_lengths,
               'BVF': bvfs}
    df = pd.DataFrame(df_dict)
    df = df.set_index('Time')

    full_df = pd.concat([full_df, df])

    if (period%10==0):
        os.system("!rm part*") # delete old part
        full_df.to_csv(f'part_{args.sim}_{period}.csv') # save new part

    print(f'Loop {period} completed: {time.time() - starttime}')
    
    del U, V, T, r, z, pres, T2, wspd, wdirs, rmol, bvfs
        
full_df.to_csv(args.output_file)