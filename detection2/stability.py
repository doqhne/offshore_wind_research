import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
from wrf import (getvar, interplevel, vertcross, 
                 vinterp, ALL_TIMES, extract_global_attrs)
import glob

def get_coords(lon,lat,POIlon,POIlat):
    '''
    Input: model lons and lats and the POI lons and lats
    '''
    km_lon = 100
    km_lat = 111 # 111 km per degree of latitude
    # Find closest grid point to desired coordinates
    dist = np.sqrt(np.square(abs(km_lon*(lon.values[:,:] - POIlon)))
                        + np.square(abs(km_lat*(lat.values[:,:] - POIlat))))
    flag = np.where(dist == np.min(dist))
    POI = [int(flag[0]), int(flag[1])]
    return(POI)

def classify_stability(arr):
    '''
    Map -1, 0, 1 to the three different stability classifications: unstable, neutral, stable
    '''
    
    arr[np.where((arr<0) & (arr>-1000))] = -1
    arr[np.where((arr>1000) | (arr<-1000))] = 0
    arr[np.where((arr>0) & (arr<1000))] = 1
    
    return arr

def plot_stab(start, stop, ax):
    start = np.where(dates == start)[0][0]
    stop = np.where(dates == stop)[0][0]
    s = 30

    ax.scatter(dates.values[start:stop], (wrf_stab)[start:stop], label='WRF', s=s, marker=2)
    ax.scatter(dates.values[start:stop], (calc_stab)[start:stop], label='calculated', s=s, marker=3)

    ax.tick_params(axis='x', labelrotation=-50)
    ax.set_xlabel('Time')
    ax.set_yticks(np.arange(-1, 2, 1))
    ax.set_yticklabels(['unstable', 'neutral', 'stable'])
    ax.legend(loc='center right', bbox_to_anchor=(0.9,0.78))

def make_plot2(th, ws, wd, T, z, m, d, h, mi, wrf_stab, calc_stab, start, stop):
    wrf_stab = stab_dict[wrf_stab]
    calc_stab = stab_dict[calc_stab]

    fig = plt.figure(figsize=(15, 8))

    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])
    ax0 = fig.add_subplot(gs[0, 0]) # th
    ax1 = fig.add_subplot(gs[0, 1]) # temp
    ax2 = fig.add_subplot(gs[0, 2]) # ws
    ax3 = fig.add_subplot(gs[0, 3]) # wd
    ax4 = fig.add_subplot(gs[1, :]) # stability timeseries
    
    ax0.plot(th, z)
    ax1.plot(T, z)
    ax2.plot(ws, z)
    ax3.scatter(wd, z)
    plot_stab(start, stop, ax4)
    
    ax0.set_xlabel('Potential temperature')
    ax1.set_xlabel('Temperature')
    ax2.set_xlabel('Wind speed')
    ax3.set_xlabel('Wind direction')

    ax0.set_ylabel('Height')
    ax3.set_xticks(np.arange(0, 360, 90))
    ax1.set_xticks(np.arange(280, 294, 4))
    ax2.set_xticks(np.arange(0, 10, 2))

    fig.suptitle(f'Conditions on 2019-{m:02}-{d:02} {h:02}:{mi:02}:00: {wrf_stab} (WRF), {calc_stab} (calc)', fontsize=16)
    
    wflag = wrf_stab[0].upper()
    cflag = calc_stab[0].upper()
    
    fig.savefig(f'stability_plots/s2019-{m:02}-{d:02}_{h:02}:{mi:02}_{wflag}{cflag}')
    
    plt.close()

# open datasets containing WRF RMOL values and calculated RMOL values
wrf = xr.open_dataset('RMOL.nc')
wrf = wrf.RMOL.values
calc_sept = xr.open_dataset('rmol_files/myrmol.ncRMOL_VWmid_2019_09.nc')
calc_oct = xr.open_dataset('rmol_files/rmol_oct.ncRMOL_VWmid_2019_10.nc')
calc = np.append(calc_sept.RMOL.values, calc_oct.RMOL.values)

# create a list of dates that maps to the wrf rmol values
dates = pd.date_range(start='9/1/2019', end='9/1/2020', freq='10min')[:-1]

# determine stability based on obukhov length
wrf_stab = classify_stability(1/wrf)
calc_stab = classify_stability(calc)

# shorten to just data we calculated
wrf_stab = wrf_stab[:8784]
dates = dates[:8784]

# find where stability classifications are different
dif_dates = dates[wrf_stab != calc_stab]

# find the date indices where this happens
dif_idx_long = np.in1d(dates, dif_dates).nonzero()[0]

# for naming files and finding wrf files
months_ = dif_dates.month
days_ = dif_dates.day
hours_ = dif_dates.hour
minutes_ = dif_dates.minute

# Directory with wrfouts
data_dir = "/pl/active/JKL_REAL/N_Atl/reruns_Beiter/wrfouts/nwf/2019/"

# Select data from domain 2
files = []
for i in range(len(dif_dates)):
    files = files + sorted([f for f in glob.glob(data_dir + f"/{months_[i]:02}/wrfout_d02_2019-{months_[i]:02}-{days_[i]:02}_{hours_[i]:02}:{minutes_[i]:02}*")])

print('file list created')

# determined independently: location of vw-centroid
lat_idx = 191
lon_idx = 232

stab_dict = {-1: 'unstable',
             0: 'neutral',
             1: 'stable'}

print('extracting variables...')
for chunk in range(33):
    if chunk == 32:
        D = [Dataset(f) for f in files[chunk*26:]]
        dif_idx = dif_idx_long[chunk*26:]
        months = months_[chunk*26:]
        days = days_[chunk*26:]
        hours = hours_[chunk*26:]
        minutes = minutes_[chunk*26:]
    else:
        D = [Dataset(f) for f in files[chunk*26:(chunk*26)+26]]
        dif_idx = dif_idx_long[chunk*26:(chunk*26)+26]
        months = months_[chunk*26:(chunk*26)+26]
        days = days_[chunk*26:(chunk*26)+26]
        hours = hours_[chunk*26:(chunk*26)+26]
        minutes = minutes_[chunk*26:(chunk*26)+26]
    print('D defined')
    # select variables, location
    th = getvar(D, 
                "th", 
                timeidx=ALL_TIMES, 
                method="cat").isel(south_north=lat_idx, west_east=lon_idx, bottom_top=slice(0, 34)) # potential temp
    wswd = getvar(D,
                  "uvmet_wspd_wdir", 
                  timeidx=ALL_TIMES, 
                  method="cat").isel(south_north=lat_idx, west_east=lon_idx, bottom_top=slice(0, 34)) # wspd/wdir
    z = getvar(D, 
               "z", 
               msl=False).isel(south_north=lat_idx, west_east=lon_idx, bottom_top=slice(0, 34)) # height
    T = getvar(D, 
               "tk", 
               timeidx=ALL_TIMES, 
               method="cat").isel(south_north=lat_idx, west_east=lon_idx, bottom_top=slice(0, 34)) # temp in kelvin

    ws = wswd.isel(wspd_wdir=0)
    wd = wswd.isel(wspd_wdir=1)

    # generate plots
    for i in range(26):
        # choose window for stability plot
        if (dif_idx[i] < 24):
            start = dates[0]
            stop = dates[dif_idx[i]+24]
        elif (dif_idx[i] > 8760):
            start = dif_idx[i]-24
            stop = dates[-1]
        else:
            start = dates[dif_idx[i]-24]
            stop = dates[dif_idx[i]+24]
        # generate plot showing profiles for the time
        make_plot2(th.isel(Time=i),
                  ws.isel(Time=i),
                  wd.isel(Time=i),
                  T.isel(Time=i),
                  z,
                  months[i],
                  days[i],
                  hours[i],
                  minutes[i],
                  wrf_stab[dif_idx[i]],
                  calc_stab[dif_idx[i]],
                  start,
                  stop)
    
    del th, wswd, T, z, ws, wd, D, start, stop, months, days, hours, minutes