'''
Find the wake area using the concave hull algorithm

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
https://towardsdatascience.com/outlier-detection-python-cd22e6a12098
'''

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull, convex_hull_plot_2d
import glob
import utils
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
pd.options.mode.chained_assignment = None
# from matplotlib.ticker import PercentFormatter
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from metpy.calc import wind_components
from metpy.units import units
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
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

def get_hgt_arr(ds, lat_idx=200, lon_idx=200):
    '''Returns an array of heights at a given location'''
    PH = utils.destagger(ds['PH'], 1)
    PH = PH.rename({'bottom_top_stag': 'bottom_top'})
    PH = PH.sel(south_north=lat_idx, west_east=lon_idx)

    PHB = utils.destagger(ds['PHB'], 1)
    PHB = PHB.rename({'bottom_top_stag': 'bottom_top'})
    PHB = PHB.sel(south_north=lat_idx, west_east=lon_idx)

    HGT = ds.HGT.sel(south_north=lat_idx, west_east=lon_idx)
    z = np.array((PH+PHB)/9.81-HGT)
    
    return z

def calc_dist(x1, x2, y1, y2):
    '''Find the distance between two lat/lon points in km'''
    x = (x2-x1) *85 # convert longitude to km
    y = (y2-y1) *111 # convert latitude to km
    dist = np.sqrt(x**2 + y**2)
    
    return dist

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

def in_hull(points, x):
    '''determine if point x lies in the cloud of points'''
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def get_flag(df):
    '''flag if there are very few points or if very few are in the wind farm'''
    # see how many points are in the wind farm
    in_wf = []
    for i in range(len(df)):
        in_wf.append(in_hull(la_points, (df.iloc[i].lon, df.iloc[i].lat)))
    pct = (len(df[in_wf]) / len(df)) *100
    
    # if there are fewer than 500 points
    if len(df)<500:
        return 1
    elif pct<15: 
        return 1
    else:
        return 0
    
def wake_distance(hull_points, ref=(-70.59, 40.95)):
    '''Find the distance between the reference point and the array of points. 
        return the maximum distance'''
    distances = []
    for p in hull_points:
        distances.append(geopy.distance.distance(ref, p).km)
        
    return np.array(distances).max()

def gen_concave_hull(time, thresh=0.1, save_plot=False):
    '''Removes outlier wake points, apply concave_hull method, compute area in km^2'''

    # read in data
    nwf_ds = xr.open_dataset(nwf_files_stable[time])
    la_ds = xr.open_dataset(la_files_stable[time])
    
    print(time, nwf_files_stable[time]) # see what time this is for

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

    # only look at region of interest and with a wspd deficit of at least 0.5 m/s
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
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1
    
    # Next, start removing points that we don't think are in the wake, first we make a dataframe
    df = pd.DataFrame()
    df['lon'] = points_filt1[:, 0]
    df['lat'] = points_filt1[:, 1]
    
    # then we use the DBSCAN algorithm to cluster points together, pick the biggest clump for the wake
    model = DBSCAN(eps=0.09, min_samples=30).fit(df.values)
    df['labels'] = model.labels_ # each point is assigned a label
    mode_label = df.labels.mode().values[0] # find the most common label, assume this is the main wake
    df_filt = df[df.labels==mode_label] # filter the dataframe for points in the wake
    
    points_filt = np.array(list(zip(df_filt.lon, df_filt.lat))) # put the points in a form suitable for concave_hull algorithm
    
    # add a flag to suspicious times
    flag = get_flag(df_filt)
    
    # Now apply the concave_hull algorithm to find a reasonable border around these points
    idxes = concave_hull_indexes(points_filt, length_threshold=thresh)
    hull_lon = concave_hull(points_filt, length_threshold=thresh)[:, 0]
    hull_lat = concave_hull(points_filt, length_threshold=thresh)[:, 1]
    
    # find the maximum distance from ONEcent and the hull points
    dist = wake_distance(list(zip(hull_lon, hull_lat)))
    
    # convert to utm coords (units meters)
    x, y = utm_proj(hull_lon, hull_lat)
    
    # make a polygon and find the area
    pgon = Polygon(zip(x, y))
    area = pgon.area / 1000**2
    print(area)
    
    if save_plot:
        # plot turbines, selected waked points, and the convexhull shape
        plt.scatter(la_turbines[1], la_turbines[0], s=1, color='grey', label='turbines')
        plt.scatter(points_filt1[:, 0], points_filt1[:, 1], s=2, alpha=0.5, color='powderblue', label='1 m/s deficit')
        plt.scatter(points_filt[:, 0], points_filt[:, 1], s=2, alpha=0.5, color='tab:blue', label='selected wake')
        for f, t in zip(idxes[:-1], idxes[1:]): 
            seg = points_filt[[f, t]]
            plt.plot(seg[:, 0], seg[:, 1], "r-", alpha=0.5)

        plt.xlim(-72.5, -67.5)
        plt.ylim(39, 42.2)
        plt.title(f'Wake area: {area:.2f} $km^{2}$ ; {nwf_files_stable[time][72:-3]}')
        lgnd = plt.legend()

        lgnd.legend_handles[0]._sizes = [60]
        lgnd.legend_handles[1]._sizes = [60]
        lgnd.legend_handles[2]._sizes = [60]

        plt.savefig(f'plots/concave_hulls/hull{nwf_files_stable[time][72:-3]}.png')
        plt.close()
        
    # we also want to output a few different pblh options to test out: ONEcent, wake area median/mean, 4km upwind    
    ONEla, ONEnwf, medla, mednwf, mnla, mnnwf, upla, upnwf = pblh_options(nwf_ds, 
                                                                          la_ds, 
                                                                          twaked_diff, 
                                                                          wdirs_stable.iloc[time]["130m wd"])
    
    return area, dist, ONEla, ONEnwf, medla, mednwf, mnla, mnnwf, upla, upnwf, flag
      
    
# read in turbine location data, ONEcent pblh data, ONEcent wdir data, ONEcent stability data
la_turbines = pd.read_csv('../turbine_locs/la100_turbines.csv', header=None, sep=' ')
la_points = np.array(list((zip(la_turbines[1], la_turbines[0]))))
wdirs = pd.read_csv('../make_wr/vwcent_wind.csv')
vw_stab = xr.open_dataset('../rmol_data/rmol_calc_vwmid3.nc')

# get a list of files for each simulation
nwf_files = utils.get_files('nwf')
la_files = utils.get_files('la_100_tke')

# subset for stable conditions only
stable = ((vw_stab.RMOL[::6]>0) & (vw_stab.RMOL[::6]<1000)).values
nwf_files_stable = np.array(nwf_files)[stable]
la_files_stable = np.array(la_files)[stable]
wdirs_stable = wdirs[stable]

# define utm projection for wake area calculations later
utm_proj = Proj(proj='utm', zone=19, ellps='WGS84')

# make lists to save wake area and pblh info into
wake_areas = []
dists = []
ONElas = []
ONEnwfs = []
medlas = []
mednwfs = []
mnlas = []
mnnwfs = []
uplas = []
upnwfs = []
flags = []

# loop through each stable time: make a plot of the wake area and save the pblh, wake info to lists
start = 4001
for time in range(start, len(nwf_files_stable)):
    area, dist, ONEla, ONEnwf, medla, mednwf, mnla, mnnwf, upla, upnwf, flag = gen_concave_hull(time, save_plot=False)
    wake_areas.append(area)
    dists.append(dist)
    ONElas.append(ONEla)
    ONEnwfs.append(ONEnwf)
    medlas.append(medla)
    mednwfs.append(mednwf)
    mnlas.append(mnla)
    mnnwfs.append(mnnwf)
    uplas.append(upla)
    upnwfs.append(upnwf)
    flags.append(flag)
    
    if time%200==0:
        df = pd.DataFrame()
        df['time'] = vw_stab.Time[::6][stable][start:time+1]
        df['wake_areas'] = wake_areas
        df['wake_dist_km'] = dists
        df['ONEcent_nwf'] = ONEnwfs
        df['ONEcent_la'] = ONElas
        df['wake_median_nwf'] = mednwfs
        df['wake_median_la'] = medlas
        df['wake_mean_nwf'] = mnnwfs
        df['wake_mean_la'] = mnlas
        df['upstream_nwf'] = upnwfs
        df['upstream_la'] = uplas
        df['flag'] = flags

        df.to_csv(f'wakes{time}.csv')
        del df
    
df = pd.DataFrame()
df['time'] = vw_stab.Time[::6][stable][start:time+1]
df['wake_areas'] = wake_areas
df['wake_dist_km'] = dists
df['ONEcent_nwf'] = ONEnwfs
df['ONEcent_la'] = ONElas
df['wake_median_nwf'] = mednwfs
df['wake_median_la'] = medlas
df['wake_mean_nwf'] = mnnwfs
df['wake_mean_la'] = mnlas
df['upstream_nwf'] = upnwfs
df['upstream_la'] = uplas
df['flag'] = flags

df.to_csv('wakes.csv')