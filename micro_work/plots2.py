import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
pd.options.mode.chained_assignment = None
from matplotlib.ticker import PercentFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import utils
    
def plot_var_vw(nwf, la100, variable, desc):
    '''
    A contour plot of the difference between nwf and la100 in vw area
    
    Parameters
    --- nwf : nwf xarray dataset
    --- la100 : la100 xarray dataset
    --- variable : name of the variable for file name labelling (e.g 'HFX')
    --- desc : description of subset of data (e.g 'stable')
    
    Returns
    --- nothing, saves a plot to plots/
    '''
    
    # calculate the difference between nwf and la100
    diff = la100[variable].mean(dim='XTIME') - nwf[variable].mean(dim='XTIME')
    print(f'diff calculated: {desc}, {variable}')
    print(diff.max().values, diff.min().values)
    
    # determine how many points we have
    num_points = len(nwf.XTIME.values)
    print(num_points)
    
    wesn_vw = [-72, -69.5, 40.3, 42.2]
    
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(), extent=wesn_vw)
    
    cmap = 'bwr'
    if variable == 'HFX':
        levels = np.arange(-3.5, 3.6, .1)
    elif variable == 'PBLH':
        levels = np.arange(-100, 101, 2)
    elif variable == 'QKE':
        levels = np.arange(-2.5, 2.55, 0.05)
    elif variable == 'wspd10':
        levels = np.arange(-2, 2.05, 0.05)
    else: #T2
        levels = np.arange(-0.5, 0.525, 0.025)
        cmap = plt.cm.bwr.reversed()
    m = ax.contourf(lons.isel(Time=0), 
                    lats.isel(Time=0), 
                    diff,
                    levels=levels,
                    cmap=cmap)

    ax.add_feature(cfeature.LAND, facecolor='grey', alpha=0.95, zorder=10)
    ax.coastlines()

    ax.set_xticks(np.arange(-71.5, -69.5, .5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(40.5, 42.1, .5), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.scatter(la_turbines[1], la_turbines[0], s=1, color='grey')

    ax.set_xlabel('Longitude [degrees]', fontsize=13)
    ax.set_ylabel('Latitude [degrees]', fontsize=13)
    ax.set_title(f'Difference in {variable} ; {desc} ; {num_points} points', fontsize=15)

    plt.colorbar(m).set_label(label=f'Difference in {variable} [LA100 - NWF]', size=13)
    
    plt.savefig(f'plots/difference_maps/vw_{variable}_{desc}_diff.png', bbox_inches='tight')

# turbine locations
la_turbines = pd.read_csv('../turbine_locs/la100_turbines.csv', header=None, sep=' ')

# one file to grab lat/lons from
f1 = xr.open_dataset('/pl/active/JKL_REAL/N_Atl/reruns_Beiter/wrfouts/nwf/2019/09/wrfout_d02_2019-09-01_00:00:00')

lats = f1.XLAT
lons = f1.XLONG

# extent of plot - from domain 2
wesn_d02 = [float(f1.XLONG.min().values),
            float(f1.XLONG.max().values)-4,
            float(f1.XLAT.min().values),
            float(f1.XLAT.max().values)]
                                
# read in the data
hfx_nwf = xr.open_dataset('out/nwf_HFX.nc')
hfx_la = xr.open_dataset('out/la_HFX.nc')
                 
pbl_nwf = xr.open_dataset('out/nwf_PBLH.nc')
pbl_la = xr.open_dataset('out/la_PBLH.nc')
                 
t2_nwf = xr.open_dataset('out/nwf_T2.nc')
t2_la = xr.open_dataset('out/la_T2.nc')

qke_nwf = xr.open_dataset('out/nwf_QKEhub.nc')
qke_la = xr.open_dataset('out/la_QKEhub.nc')

ws10_nwf = xr.open_dataset('out/nwf_ws10.nc')
ws10_la = xr.open_dataset('out/la_ws10.nc')
ws10_nwf = ws10_nwf.rename_vars({'__xarray_dataarray_variable__': 'wspd10'})
ws10_la = ws10_la.rename_vars({'__xarray_dataarray_variable__': 'wspd10'})

vwwind = pd.read_csv('../make_wr/vwcent_wind.csv')

vw_stab = xr.open_dataset('../rmol_data/rmol_calc_vwmid3.nc')

print('DATA READ IN COMPLETE')

# ----- PLOTTING -----

# make maps for:
    # 4 quadrants of wind directions
    # 3 wind speed classes
    # stability classifications

var_names = ['QKE', 'wspd10']
# for i, var in enumerate([[hfx_nwf, hfx_la], [pbl_nwf, pbl_la], [t2_nwf, t2_la], [qke_nwf, qke_la], [ws10_nwf, ws10_la]]):
for i, var in enumerate([[qke_nwf, qke_la], [ws10_nwf, ws10_la]]):
#     # sw winds
#     sw_nwf = var[0].sel(XTIME=((vwwind['130m wd']>180) & (vwwind['130m wd']<=270)).values)
#     sw_la = var[1].sel(XTIME=((vwwind['130m wd']>180) & (vwwind['130m wd']<=270)).values)
#     plot_var_vw(sw_nwf, sw_la, var_names[i], 'sw_wind')
#     del sw_nwf, sw_la

#     # nw winds
#     nw_nwf = var[0].sel(XTIME=(vwwind['130m wd']>270).values)
#     nw_la = var[1].sel(XTIME=(vwwind['130m wd']>270).values)
#     plot_var_vw(nw_nwf, nw_la, var_names[i], 'nw_wind')
#     del nw_nwf, nw_la

#     # ne winds
#     ne_nwf = var[0].sel(XTIME=(vwwind['130m wd']<=90).values)
#     ne_la = var[1].sel(XTIME=(vwwind['130m wd']<=90).values)
#     plot_var_vw(ne_nwf, ne_la, var_names[i], 'ne_wind')
#     del ne_nwf, ne_la

#     # se winds
#     se_nwf = var[0].sel(XTIME=((vwwind['130m wd']>90) & (vwwind['130m wd']<=180)).values)
#     se_la = var[1].sel(XTIME=((vwwind['130m wd']>90) & (vwwind['130m wd']<=180)).values)
#     plot_var_vw(se_nwf, se_la, var_names[i], 'se_wind')
#     del se_nwf, se_la

#     # 0-3 m/s wind at hub height (130m)
#     slow_nwf = var[0].sel(XTIME=(vwwind['130m ws']<=3).values)
#     slow_la = var[1].sel(XTIME=(vwwind['130m ws']<=3).values)
#     plot_var_vw(slow_nwf, slow_la, var_names[i], 'hh00-03')
#     del slow_nwf, slow_la

#     # 3-11 m/s wind at hub height (130m)
#     mod_nwf = var[0].sel(XTIME=((vwwind['130m ws']>3) & (vwwind['130m ws']<=11)).values)
#     mod_la = var[1].sel(XTIME=((vwwind['130m ws']>3) & (vwwind['130m ws']<=11)).values)
#     plot_var_vw(mod_nwf, mod_la, var_names[i], 'hh03-11')
#     del mod_nwf, mod_la   

#     # 11+ m/s wind at hub height (130m)
#     fast_nwf = var[0].sel(XTIME=(vwwind['130m ws']>11).values)
#     fast_la = var[1].sel(XTIME=(vwwind['130m ws']>11).values)
#     plot_var_vw(fast_nwf, fast_la, var_names[i], 'hh11+')
#     del fast_nwf, fast_la
    
    # stable conditions
    s_nwf = var[0].sel(XTIME=((vw_stab.RMOL[::6]>0) & (vw_stab.RMOL[::6]<1000)).values)
    s_la = var[1].sel(XTIME=((vw_stab.RMOL[::6]>0) & (vw_stab.RMOL[::6]<1000)).values)
    plot_var_vw(s_nwf, s_la, var_names[i], 'stable')
    del s_nwf, s_la
    
    # unstable conditions
    u_nwf = var[0].sel(XTIME=((vw_stab.RMOL[::6]<0) & (vw_stab.RMOL[::6]>-1000)).values)
    u_la = var[1].sel(XTIME=((vw_stab.RMOL[::6]<0) & (vw_stab.RMOL[::6]>-1000)).values)
    plot_var_vw(u_nwf, u_la, var_names[i], 'unstable')
    del u_nwf, u_la
    
    # neutral conditions
    n_nwf = var[0].sel(XTIME=((vw_stab.RMOL[::6]<-1000) | (vw_stab.RMOL[::6]>1000)).values)
    n_la = var[1].sel(XTIME=((vw_stab.RMOL[::6]<-1000) | (vw_stab.RMOL[::6]>1000)).values)
    plot_var_vw(n_nwf, n_la, var_names[i], 'neutral')
    del n_nwf, n_la
    