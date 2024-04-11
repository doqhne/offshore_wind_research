'''Make difference plot panels for large datasets: need to process each panel separately'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    
def plot_var_vw(nwf, la100, variable, desc, ax):
    '''
    A contour plot of the difference between nwf and la100 in vw area
    
    Parameters
    --- nwf : nwf xarray dataset
    --- la100 : la100 xarray dataset
    --- variable : name of the variable for file name labelling (e.g 'HFX')
    --- desc : description of subset of data (e.g 'stable')
    --- ax : the subplot axes to use
    
    Returns
    --- m, mappable to use for the figure colorbar, a figure is also saved
    '''
    
   
    # calculate the difference between nwf and la100
    diff = la100[variable].mean(dim='XTIME') - nwf[variable].mean(dim='XTIME')
    print(f'--- diff calculated: {desc}, {variable}, {diff.min().values:.3f}, {diff.max().values:.3f}')
    
    # determine how many points we have
    num_points = len(nwf.XTIME.values)

    wesn_vw = [-72, -69.5, 40.3, 42.2]
    
    cmap = plt.cm.bwr
    
    if variable == 'HFX':
        levels = np.arange(-3.5, 3.6, .5)
    elif variable == 'PBLH':
        levels = np.arange(-120, 121, 20)
    elif variable == 'QKEhub':
        levels = np.arange(-2.5, 2.55, 0.25) # hub
    elif variable == 'QKEsfc':
        levels = np.arange(-0.5, 0.55, 0.05) # surface
    elif variable == 'hub_wspd':
        levels = np.arange(-4, 4.5, 0.5)
    else: #T2
        levels = np.arange(-0.25, 0.26, 0.025)
    
    # plot turbines
    ax.scatter(la_turbines[1], la_turbines[0], s=3, color='grey')
    
    m = ax.contourf(lons.isel(Time=0), 
                    lats.isel(Time=0), 
                    diff,
                    cmap=cmap,
                    levels=levels,
                    alpha=0.8)

    ax.add_feature(cfeature.LAND, facecolor='grey', zorder=10)
    ax.coastlines()

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    ax.set_title(f'{variable}; {desc}', fontsize=13)
    
    return m

def wd_panel(nwf, la, var_name):
    '''Plot a 4 panel plot for each wind direction for a given variable'''
    
    fig, axs = plt.subplots(nrows=2, ncols=2, 
                    sharey=True, sharex=True, 
                    subplot_kw={'projection': ccrs.PlateCarree(), 'extent': wesn_vw},
                    figsize=(10, 8))

    ax0 = axs[0][0]
    ax1 = axs[0][1]
    ax2 = axs[1][0]
    ax3 = axs[1][1]

    # ne wind
    nwf0 = nwf.sel(XTIME=(vwwind['130m wd']>270).values)
    la0 = la.sel(XTIME=(vwwind['130m wd']>270).values)
    m = plot_var_vw(nwf0, la0, var_name, 'NW', ax0)
    del nwf0, la0

    # nw wind
    nwf1 = nwf.sel(XTIME=(vwwind['130m wd']<=90).values)
    la1 = la.sel(XTIME=(vwwind['130m wd']<=90).values)
    m = plot_var_vw(nwf1, la1, var_name, 'NE', ax1)
    del nwf1, la1

    # sw wind
    nwf2 = nwf.sel(XTIME=((vwwind['130m wd']>180) & (vwwind['130m wd']<=270)).values)
    la2 = la.sel(XTIME=((vwwind['130m wd']>180) & (vwwind['130m wd']<=270)).values)
    m = plot_var_vw(nwf2, la2, var_name, 'SW', ax2)
    del nwf2, la2

    # se wind
    nwf3 = nwf.sel(XTIME=((vwwind['130m wd']>90) & (vwwind['130m wd']<=180)).values)
    la3 = la.sel(XTIME=((vwwind['130m wd']>90) & (vwwind['130m wd']<=180)).values)
    m = plot_var_vw(nwf3, la3, var_name, 'SE', ax3)
    del nwf3, la3

    for ax in [ax0, ax2]:
        ax.set_ylabel('Latitude [degrees]', fontsize=13)
        ax.set_yticks(np.arange(40.5, 42.1, .5), crs=ccrs.PlateCarree())
        ax.set_yticklabels(np.arange(40.5, 42.1, .5), fontsize=12)
        ax.yaxis.set_major_formatter(lat_formatter)

    for ax in [ax2, ax3]:
        ax.set_xlabel('Longitude [degrees]', fontsize=13)
        ax.set_xticks(np.arange(-71.5, -69.5, .5), crs=ccrs.PlateCarree())
        ax.set_xticklabels(np.arange(-71.5, -69.5, .5), fontsize=12)
        ax.xaxis.set_major_formatter(lon_formatter)

    fig.subplots_adjust(right=0.8, wspace=0.04, hspace=0)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.67])
    cbar = fig.colorbar(m, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(label='LA100 - NWF', size=13)
    
    fig.savefig(f'plots/difference_maps/panels/{var_name}_wdir_panel.png', bbox_inches='tight')
    
    plt.close()
    
def stablity_panel(nwf, la, var_name):
    '''create a 3 panel plot for each stability class for a given variable'''
    
    fig, axs = plt.subplots(nrows=1, ncols=3, 
                        sharey=True, 
                        subplot_kw={'projection': ccrs.PlateCarree(), 'extent': wesn_vw},
                        figsize=(20, 5))
    ax0 = axs[0]
    ax1 = axs[1]
    ax2 = axs[2]

    # stable
    nwf0 = nwf.sel(XTIME=((vw_stab.RMOL[::6]>0) & (vw_stab.RMOL[::6]<500)).values)
    la0 = la.sel(XTIME=((vw_stab.RMOL[::6]>0) & (vw_stab.RMOL[::6]<500)).values)
    m = plot_var_vw(nwf0, la0, var_name, 'stable', ax0)
    del nwf0, la0
    
    # neutral
    nwf1 = nwf.sel(XTIME=((vw_stab.RMOL[::6]<-500) | (vw_stab.RMOL[::6]>500)).values)
    la1 = la.sel(XTIME=((vw_stab.RMOL[::6]<-500) | (vw_stab.RMOL[::6]>500)).values)
    m = plot_var_vw(nwf1, la1, var_name, 'neutral', ax1)
    del nwf1, la1
    
    # unstable
    nwf2 = nwf.sel(XTIME=((vw_stab.RMOL[::6]<0) & (vw_stab.RMOL[::6]>-500)).values)
    la2 = la.sel(XTIME=((vw_stab.RMOL[::6]<0) & (vw_stab.RMOL[::6]>-500)).values)
    m = plot_var_vw(nwf2, la2, var_name, 'unstable', ax2)
    del nwf2, la2
    
    ax0.set_ylabel('Latitude [degrees]', fontsize=14)
    ax0.set_yticks(np.arange(40.5, 42.1, .5), crs=ccrs.PlateCarree())
    ax0.set_yticklabels(np.arange(40.5, 42.1, .5), fontsize=12)
    ax0.yaxis.set_major_formatter(lat_formatter)

    for ax in axs:
        ax.set_xlabel('Longitude [degrees]', fontsize=14)
        ax.set_xticks(np.arange(-71.5, -69.5, .5), crs=ccrs.PlateCarree())
        ax.set_xticklabels(np.arange(-71.5, -69.5, .5), fontsize=12)
        ax.xaxis.set_major_formatter(lon_formatter)

    fig.subplots_adjust(right=0.8, wspace=0.05)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.015, 0.67])
    cbar = fig.colorbar(m, cax=cbar_ax, cmap='bwr', label='LA100 - NWF')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(label='LA100 - NWF', size=14)
    
    fig.savefig(f'plots/difference_maps/panels/{var_name}_stability_panel.png', bbox_inches='tight')
    
    plt.close()
    
def wspd_panel(nwf, la, var_name):
    '''create plots by hub height wind speed for a given variable, stable SW winds only '''
    
    fig, axs = plt.subplots(nrows=1, ncols=3, 
                        sharey=True, 
                        subplot_kw={'projection': ccrs.PlateCarree(), 'extent': wesn_vw},
                        figsize=(20, 5))
    ax0 = axs[0]
    ax1 = axs[1]
    ax2 = axs[2]

    # 0-3 m/s
#     nwf0 = nwf.sel(XTIME=(vwwind['130m ws']<=3).values)
#     la0 = la.sel(XTIME=(vwwind['130m ws']<=3).values)
    nwf0 = nwf.sel(XTIME=((vwwind['130m ws']<=3) & 
                          (vwwind['130m wd']>180) & (vwwind['130m wd']<=270) & 
                          (vw_stab.RMOL[::6].values>0) & (vw_stab.RMOL[::6].values<1000)).values)
    la0 = la.sel(XTIME=((vwwind['130m ws']<=3) & 
                        (vwwind['130m wd']>180) & (vwwind['130m wd']<=270) & 
                        (vw_stab.RMOL[::6].values>0) & (vw_stab.RMOL[::6].values<1000)).values)
    m = plot_var_vw(nwf0, la0, var_name, '0-3 m s$^{-1}$', ax0)
    del nwf0, la0
    
    # 3-11 m/s
#     nwf1 = nwf.sel(XTIME=((vwwind['130m ws']>3) & (vwwind['130m ws']<=11)).values)
#     la1 = la.sel(XTIME=((vwwind['130m ws']>3) & (vwwind['130m ws']<=11)).values)
    nwf1 = nwf.sel(XTIME=((vwwind['130m ws']>3) & (vwwind['130m ws']<=11) & 
                          (vwwind['130m wd']>180) & (vwwind['130m wd']<=270) & 
                          (vw_stab.RMOL[::6].values>0) & (vw_stab.RMOL[::6].values<1000)).values)
    la1 = la.sel(XTIME=((vwwind['130m ws']>3) & (vwwind['130m ws']<=11) & 
                        (vwwind['130m wd']>180) & (vwwind['130m wd']<=270) & 
                        (vw_stab.RMOL[::6].values>0) & (vw_stab.RMOL[::6].values<1000)).values)
    m = plot_var_vw(nwf1, la1, var_name, '3-11 m s$^{-1}$', ax1)
    del nwf1, la1
    
    # 11+ m/s
#     nwf2 = nwf.sel(XTIME=(vwwind['130m ws']>11).values)
#     la2 = la.sel(XTIME=(vwwind['130m ws']>11).values)
    nwf2 = nwf.sel(XTIME=((vwwind['130m ws']>11) & 
                          (vwwind['130m wd']>180) & (vwwind['130m wd']<=270) & 
                          (vw_stab.RMOL[::6].values>0) & (vw_stab.RMOL[::6].values<1000)).values)
    la2 = la.sel(XTIME=((vwwind['130m ws']>11) & 
                        (vwwind['130m wd']>180) & (vwwind['130m wd']<=270) & 
                        (vw_stab.RMOL[::6].values>0) & (vw_stab.RMOL[::6].values<1000)).values)
    m = plot_var_vw(nwf2, la2, var_name, '11+ m s$^{-1}$', ax2)
    del nwf2, la2
    
    ax0.set_ylabel('Latitude [degrees]', fontsize=14)
    ax0.set_yticks(np.arange(40.5, 42.1, .5), crs=ccrs.PlateCarree())
    ax0.set_yticklabels(np.arange(40.5, 42.1, .5), fontsize=12)
    ax0.yaxis.set_major_formatter(lat_formatter)

    for ax in axs:
        ax.set_xlabel('Longitude [degrees]', fontsize=14)
        ax.set_xticks(np.arange(-71.5, -69.5, .5), crs=ccrs.PlateCarree())
        ax.set_xticklabels(np.arange(-71.5, -69.5, .5), fontsize=12)
        ax.xaxis.set_major_formatter(lon_formatter)

    fig.subplots_adjust(right=0.8, wspace=0.05)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.015, 0.67])
    cbar = fig.colorbar(m, cax=cbar_ax, cmap='bwr', label='LA100 - NWF')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(label='LA100 - NWF', size=14)

    fig.savefig(f'plots/difference_maps/panels/{var_name}_wspd_panel.png', bbox_inches='tight')
    
    plt.close()

# ---- READ IN DATA ----

# turbine locations
la_turbines = pd.read_csv('../turbine_locs/la100_turbines.csv', header=None, sep=' ')

# one file to grab lat/lons from
f1 = xr.open_dataset('/pl/active/JKL_REAL/N_Atl/reruns_Beiter/wrfouts/nwf/2019/09/wrfout_d02_2019-09-01_00:00:00')
                               
# nwf and la100 files for each variable
qkehub_nwf = xr.open_dataset('out/nwf_QKEhub.nc')
qkehub_la = xr.open_dataset('out/la_QKEhub.nc')
qkehub_nwf = qkehub_nwf.rename_vars({'QKE': 'QKEhub'})
qkehub_la = qkehub_la.rename_vars({'QKE': 'QKEhub'})

qkesfc_nwf = xr.open_dataset('out/nwf_QKE.nc')
qkesfc_la = xr.open_dataset('out/la_QKE.nc')
qkesfc_nwf = qkesfc_nwf.rename_vars({'QKE': 'QKEsfc'})
qkesfc_la = qkesfc_la.rename_vars({'QKE': 'QKEsfc'})

hfx_nwf = xr.open_dataset('out/nwf_HFX.nc')
hfx_la = xr.open_dataset('out/la_HFX.nc')
                 
pbl_nwf = xr.open_dataset('out/nwf_PBLH.nc')
pbl_la = xr.open_dataset('out/la_PBLH.nc')
                 
t2_nwf = xr.open_dataset('out/nwf_T2.nc')
t2_la = xr.open_dataset('out/la_T2.nc')

wshub_nwf = xr.open_dataset('out/nwf_hubws.nc')
wshub_la = xr.open_dataset('out/la_hubws.nc')
wshub_nwf = wshub_nwf.rename_vars({'__xarray_dataarray_variable__': 'hub_wspd'})
wshub_la = wshub_la.rename_vars({'__xarray_dataarray_variable__': 'hub_wspd'})

# hub height wind speed, wind direction, and atmospheric stability at ONE centroid
vwwind = pd.read_csv('../make_wr/vwcent_wind.csv')
vw_stab = xr.open_dataset('../rmol_data/rmol_calc_vwmid3.nc')


# ---- DEFINE VARIABLES ----

lats = f1.XLAT
lons = f1.XLONG

# extent of plot from domain 2
wesn_vw = [-72, -69.5, 40.3, 42.2]
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()

# make lists of each dataset and variable name
var_names = np.array(['QKEhub', 'QKEsfc', 'HFX', 'PBLH', 'T2', 'hub_wspd'])
data_list = [[qkehub_nwf, qkehub_la], [qkesfc_nwf, qkesfc_la], [hfx_nwf, hfx_la], [pbl_nwf, pbl_la], [t2_nwf, t2_la], [wshub_nwf, wshub_la]]


# ---- MAKE PLOTS FOR EACH VARIABLE ----

# subset just variables we want
idxs = np.array([1])
var_names = [var_names[i] for i in idxs]
data_list = [data_list[i] for i in idxs]

# make panels for each variable
for i, var in enumerate(data_list):
    print(f'{var_names[i]}')
    print('- Generating stability panel')
    stablity_panel(var[0], var[1], var_names[i])
#     print('- Generating wind direction panel')
#     wd_panel(var[0], var[1], var_names[i], h=h)
    print('- Generating wind speed panel')
    wspd_panel(var[0], var[1], var_names[i])