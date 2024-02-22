#!/usr/bin/env python
'''
Find the wake area using the concave hull algorithm

Flagging:
- if the number of total waked points is less than 4, no plot is created as we can't create a polygon
- if there are no labels with at least 5% of points in the wind farm, no plots is created as no wake can be identified
- if the number of total waked points is less than 200, flag
- if the percentage of points in the wind farm that are waked is less than 40%, flag
- if the ratio selected wake points / total waked points is less than 0.2, flag
'''

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import glob
import utils
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
pd.options.mode.chained_assignment = None
from metpy.calc import wind_components
from metpy.units import units
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, Point
from pyproj import Proj
from scipy.optimize import linprog
from concave_hull import concave_hull, concave_hull_indexes
import geopy
import geopy.distance

def calc_windspeed(ds):
    '''Calculate the wind speed using U and V'''
    U = utils.destagger(ds.U, 3)
    U = U.rename({'west_east_stag': 'west_east'})
    V = utils.destagger(ds.V, 2)
    V = V.rename({'south_north_stag': 'south_north'})
    wspd = np.sqrt((U**2)+(V**2))
    
    return wspd

def calc_dist(x1, x2, y1, y2):
    '''Find the distance between two lat/lon points in km'''
    x = (x2-x1) *85 # convert longitude to km
    y = (y2-y1) *111 # convert latitude to km
    dist = np.sqrt(x**2 + y**2)
    
    return dist

def in_cloud(point):
    '''Determine if point is in the turbine cloud'''
    return point.within(turb_pgon)

def get_pct2(df):
    '''find the percentage of points that are in the wind farm'''
    in_wf = df.apply(lambda row: in_cloud(Point(row['lon'],row['lat'])), axis=1)
    pct = (len(df[in_wf]) / len(df)) *100
    return pct, len(df[in_wf]

def wake_distance(hull_points, ref=(-70.59, 40.95)):
    '''Find the distance between the reference point and the array of points. 
        return the maximum distance'''
    distances = []
    for p in hull_points:
        distances.append(geopy.distance.distance(ref, p).km)
        
    return np.array(distances).max()

def pblh_options(nwf, la, twaked_diff, wdir):
    '''Find a few different pbl heights to compare'''
    
    # find pblh at ONEcent
    lon_idx, lat_idx = utils.find_lat_lon_idx(nwf.XLONG.isel(Time=0).values, 
                                              nwf.XLAT.isel(Time=0).values, 
                                              -70.59, 40.95)
    ONEla = la.PBLH.isel(Time=0, south_north=lat_idx, west_east=lon_idx).values
    ONEnwf = nwf.PBLH.isel(Time=0, south_north=lat_idx, west_east=lon_idx).values
    
    # find median pblh for waked area
    medla = la.PBLH.isel(Time=0).where(twaked_diff.fillna(False), np.nan).median().values
    mednwf = nwf.PBLH.isel(Time=0).where(twaked_diff.fillna(False), np.nan).median().values
    
    # find mean pblh for waked area
    mnla = la.PBLH.isel(Time=0).where(twaked_diff.fillna(False), np.nan).mean().values
    mnnwf = nwf.PBLH.isel(Time=0).where(twaked_diff.fillna(False), np.nan).mean().values
    
    # find pblh 4km upwind of wind direction at onecent
    start = geopy.Point(40.95, -70.59)
    d = geopy.distance.distance(kilometers=4)
    final = d.destination(point=start, bearing=wdir)
    lon_idx, lat_idx = utils.find_lat_lon_idx(nwf.XLONG.isel(Time=0).values, 
                                              nwf.XLAT.isel(Time=0).values, 
                                              final[1], final[0])
    upla = la.PBLH.isel(Time=0, south_north=lat_idx, west_east=lon_idx).values
    upnwf = nwf.PBLH.isel(Time=0, south_north=lat_idx, west_east=lon_idx).values
    
    return ONEla, ONEnwf, medla, mednwf, mnla, mnnwf, upla, upnwf

def select_label(df):
    '''Choose the best label for the wake'''
    # all options for the labels
    labels = df.labels.unique()
    
    # remove the noise label: -1
    labels_nonoise = labels[labels>-1]
    if len(labels_nonoise)==0:
        return None
    
    # find the percentage of points in the wind farm for each label
    pcts = []
    for l in labels_nonoise:
        pct = get_pct2(df[df.labels==l])
        pcts.append(pct)
        
    # select only labels with 15% or more points in the wind farm
    labels_inwf = labels_nonoise[np.array(pcts)>5]
    if len(labels_inwf)==0:
        return None
    
    # find the label with the most points total
    num_points = []
    for l in labels_inwf:
        num_points.append(df.labels.value_counts()[l])
        
    
    return labels_inwf[np.array(num_points).argmax()]

def plot_concave_hull(time, thresh=0.1, eps=0.1, save_plot=False):
    flag = 0
    # read in data
    nwf_ds = xr.open_dataset(nwf_files_stable[time])
    la_ds = xr.open_dataset(la_files_stable[time])
    print(time, nwf_files_stable[time])

    #calculate hub height wind speed
    nwf_wspd = calc_windspeed(nwf_ds).isel(bottom_top=11, Time=0)
    la_wspd = calc_windspeed(la_ds).isel(bottom_top=11, Time=0)

    # calculate the wind speed difference
    wspd_diff = la_wspd - nwf_wspd
    
    # nan all values not in the region of interest
    ext_bool = ((nwf_ds.XLAT.isel(Time=0)>39) & 
                (nwf_ds.XLAT.isel(Time=0)<42.3) & 
                (nwf_ds.XLONG.isel(Time=0)>-72.5) & 
                (nwf_ds.XLONG.isel(Time=0)<-67.5))

    # only look at region of interest and with a wspd deficit of at least 1 m/s
    trimmed_diff = wspd_diff.where(ext_bool)
    twaked_diff = trimmed_diff.where(trimmed_diff<-1)

    # mask latitude and longitude grids based on the wind speed deficit array
    lats_filt = nwf_ds.XLAT.isel(Time=0).where(twaked_diff.fillna(False), np.nan)
    lons_filt = nwf_ds.XLONG.isel(Time=0).where(twaked_diff.fillna(False), np.nan)

    # unravel 2D lat/lon grids and put in (lon, lat) form, then remove all points containing nan
    points = list(zip(lons_filt.values.ravel(), lats_filt.values.ravel()))
    points_filt1 = np.array([t for t in points if not any(np.isnan(x) for x in t)])
    
    # avoid issues with the concave algorithm or polygon creation
    if len(points_filt1)<4:
        print('error')
        return np.nan, np.nan, 1
    elif len(points_filt1)<200:
        flag = 1
    
    # First we make a dataframe
    df = pd.DataFrame()
    df['lon'] = points_filt1[:, 0]
    df['lat'] = points_filt1[:, 1]

    # then we use the DBSCAN algorithm to cluster points together, pick the biggest clump for the wake
    model = DBSCAN(eps=eps, min_samples=30).fit(df.values)
    df['labels'] = model.labels_ # each point is assigned a label
    
    pcts = []
    for l in df.labels.unique():
        pct = get_pct2(df[df.labels==l])[0]
        pcts.append(pct)
        
#     if np.array(pcts).max()<20:
#         flag = 1
        
    # select the label with the most points in the wind farm
    label = select_label(df)
    if label==None:
        return np.nan, np.nan, 1
                    
    # Flagging: how much of the wind farm is waked? If less than 40%, flag
    all_points = list(zip(nwf_ds.XLONG.isel(Time=0).values.ravel(), 
                          nwf_ds.XLAT.isel(Time=0).values.ravel()))
    all_inwf = [] 
    for p in all_points:
        all_inwf.append(in_cloud(Point(p)))
    tpoints = np.array(all_points)[all_inwf]
    if (get_pct2(df[df.labels==label])[1] / len(tpoints)) < 0.4:
        flag = 1
    
    df_filt = df[df.labels==label] # filter the dataframe for points in the wake
    points_filt = np.array(list(zip(df_filt.lon, df_filt.lat))) # put the points in a form suitable for concave_hull algorithm
    if (len(points_filt) / len(points_filt1)) < 0.15:
        flag = 1
    
    # Now apply the concave_hull algorithm to find a reasonable border around these points
    idxes = concave_hull_indexes(points_filt, length_threshold=thresh)
    hull_lon = concave_hull(points_filt, length_threshold=thresh)[:, 0]
    hull_lat = concave_hull(points_filt, length_threshold=thresh)[:, 1]
    
    x, y = utm_proj(hull_lon, hull_lat) # convert to cartesian coordinates
    
    # make a polygon and find the area
    pgon = Polygon(zip(x, y))
    area = pgon.area / 1000**2 # find the area, then convert from m^2 to km^2
    dist = wake_distance(list(zip(hull_lon, hull_lat)))
    print(area, dist)
    
    if save_plot:
        if flag == 1:
            title_color = 'r'
        else:
            title_color = 'k'
        plt.scatter(la_turbines.longitude, la_turbines.latitude, s=1, color='grey', label='turbines')
        plt.scatter(points_filt1[:, 0], points_filt1[:, 1], s=2, alpha=0.5, color='powderblue', label='1 m/s deficit')
        plt.scatter(points_filt[:, 0], points_filt[:, 1], s=2, alpha=0.5, color='tab:blue', label='selected wake')
        for f, t in zip(idxes[:-1], idxes[1:]): 
            seg = points_filt[[f, t]]
            plt.plot(seg[:, 0], seg[:, 1], "r-", alpha=0.5)

        plt.xlim(-72.5, -67.5)
        plt.ylim(39, 42.2)
        plt.title(f'Wake area: {area:.0f} $km^{2}$ ; dist: {dist:.1f}km ; {nwf_files_stable[time][72:-3]} ; time: {time}',
                  color=title_color)
        lgnd = plt.legend()

        lgnd.legend_handles[0]._sizes = [60]
        lgnd.legend_handles[1]._sizes = [60]
        lgnd.legend_handles[2]._sizes = [60]
        
        plt.savefig(f'plots/concave_hulls2/hull{time}_{nwf_files_stable[time][72:-3]}.png')
        plt.close()
        
    return area, dist, flag
      
    
# read in turbine location data, ONEcent pblh data, ONEcent wdir data, ONEcent stability data
la_turbines = pd.read_csv('../turbine_locs/la100_turbines.csv', header=None, sep=' ', names=['latitude', 'longitude', 'type'])
wdirs = pd.read_csv('../make_wr/vwcent_wind.csv')
vw_stab = xr.open_dataset('../rmol_data/rmol_calc_vwmid3.nc')

# set up variables we will need
nwf_files = utils.get_files('nwf')
la_files = utils.get_files('la_100_tke')
utm_proj = Proj(proj='utm', zone=19, ellps='WGS84')
la_turbines = la_turbines[(la_turbines.latitude>40)&(la_turbines.longitude>-72)&(la_turbines.longitude<-69.5)]
la_points = np.array(list((zip(la_turbines['longitude'], la_turbines['latitude']))))
turb_pgon = Polygon(concave_hull(la_points, length_threshold=0.05))

# subset for stable conditions only
stable = ((vw_stab.RMOL[::6]>0) & (vw_stab.RMOL[::6]<1000)).values
nwf_files_stable = np.array(nwf_files)[stable]
la_files_stable = np.array(la_files)[stable]
wdirs_stable = wdirs[stable]

# make lists to save wake area and pblh info into
wake_areas = []
dists = []
# ONElas = []
# ONEnwfs = []
# medlas = []
# mednwfs = []
# mnlas = []
# mnnwfs = []
# uplas = []
# upnwfs = []
flags = []

# loop through each stable time: make a plot of the wake area and save the pblh, wake info to lists
start = 0
for time in range(start, len(nwf_files_stable)):
#     area, dist, ONEla, ONEnwf, medla, mednwf, mnla, mnnwf, upla, upnwf, flag = gen_concave_hull(time, save_plot=False)
    if time%5==0:
        save_plot = True
    else:
        save_plot = False
    area, dist, flag = plot_concave_hull(time, save_plot=save_plot)
    wake_areas.append(area)
    dists.append(dist)
#     ONElas.append(ONEla)
#     ONEnwfs.append(ONEnwf)
#     medlas.append(medla)
#     mednwfs.append(mednwf)
#     mnlas.append(mnla)
#     mnnwfs.append(mnnwf)
#     uplas.append(upla)
#     upnwfs.append(upnwf)
    flags.append(flag)
    
    if time%500==0:
        df = pd.DataFrame()
        df['time'] = vw_stab.Time[::6][stable][start:time+1]
        df['wake_areas'] = wake_areas
        df['wake_dist_km'] = dists
#         df['ONEcent_nwf'] = ONEnwfs
#         df['ONEcent_la'] = ONElas
#         df['wake_median_nwf'] = mednwfs
#         df['wake_median_la'] = medlas
#         df['wake_mean_nwf'] = mnnwfs
#         df['wake_mean_la'] = mnlas
#         df['upstream_nwf'] = upnwfs
#         df['upstream_la'] = uplas
        df['flag'] = flags

        df.to_csv(f'wakes2_{time}.csv')
        del df
    
df = pd.DataFrame()
df['time'] = vw_stab.Time[::6][stable][start:time+1]
df['wake_areas'] = wake_areas
df['wake_dist_km'] = dists
# df['ONEcent_nwf'] = ONEnwfs
# df['ONEcent_la'] = ONElas
# df['wake_median_nwf'] = mednwfs
# df['wake_median_la'] = medlas
# df['wake_mean_nwf'] = mnnwfs
# df['wake_mean_la'] = mnlas
# df['upstream_nwf'] = upnwfs
# df['upstream_la'] = uplas
df['flag'] = flags

df.to_csv('wakes2.csv')