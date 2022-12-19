'''
Functions for various calculations related to LLJ detection

    * destagger : Destagger for a given dimension
    * find_lat_lon_idx : Find the index of the latitude and longitude 
                         closest to inputted values
    * find_idx : Find index and value of nearest value in array
    * find_shear_veer : Find shear and veer between 2 heights
    * find_veer : Find the veer given U and V at 2 different heights
    * find_classification : Find the LLJ calculation according to Vanderwende
    * isLLJ : Determines if an LLJ has occurred
'''

import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units


def destagger(var, stagger_dim):
    '''
    From wrf-python 
    https://github.com/NCAR/wrf-python/blob/b40d1d6e2d4aea3dd2dda03aae18e268b1e9291e/src/wrf/destag.py 
    '''
    var_shape = var.shape
    num_dims = var.ndim
    stagger_dim_size = var_shape[stagger_dim]

    full_slice = slice(None)
    slice1 = slice(0, stagger_dim_size - 1, 1)
    slice2 = slice(1, stagger_dim_size, 1)

    dim_ranges_1 = [full_slice] * num_dims
    dim_ranges_2 = [full_slice] * num_dims

    dim_ranges_1[stagger_dim] = slice1
    dim_ranges_2[stagger_dim] = slice2

    result = .5*(var[tuple(dim_ranges_1)] + var[tuple(dim_ranges_2)])

    return result

def find_lat_lon_idx(lons, lats, lon, lat):
    '''
    Finds the index of the latitude and longitude closest to inputted values
    
    Returns tuple with indexes: (lat, lon)
    '''
    # conversions
    km_lon = 85
    km_lat = 111

    # Find the closest model grid cell to the lidar (or whatever location you're interested in)
    dist = np.sqrt(np.square(abs(km_lon*(lons-lon)))
                  + np.square(abs(km_lat*(lats-lat))))
    loc = np.where(dist==np.min(dist))
    
    return int(loc[0]), int(loc[1])

def find_idx(array, value):
    '''Find index and value of nearest value in array
    * Only use this for heights!
    '''
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_shear_veer(h0, h1, wspds, heights, U, V):
    '''
    Find the wind shear and veer using the two inputted heights
    
    Parameters
    ----------
    h0 : The lower height, in meters
    
    h1 : The upper height, in meters
    
    wspds : The wind speeds at a certain time
    
    heights : The heights at a certain time
    
    U : The u component of wind at a certain time
    
    V : The v component of the wind at a certain time
    
    Returns
    -------
    tuple of form: (shear, veer, dz)
    '''

    # Find index of height h0 and h1
    h0_idx = find_idx(heights, h0)[0]
    h1_idx = find_idx(heights, h1)[0]
    
    # Find windspeed at each height
    h0_wspd = wspds.sel(bottom_top=h0_idx).values
    h1_wspd = wspds.sel(bottom_top=h1_idx).values
    
    # Find shear
    shear = h1_wspd - h0_wspd
    
    # Find veer
    veer = find_veer(U.sel(bottom_top=h0_idx), # lower U
                     V.sel(bottom_top=h0_idx), # lower V
                     U.sel(bottom_top=h1_idx), # upper U
                     V.sel(bottom_top=h1_idx)) # upper V
    
    # Find dz
    #dz = find_idx(heights, h1)[1] - find_idx(heights, h1)[1]
    dz = heights[h1_idx] - heights[h0_idx]
    
    return (shear, veer, dz)

def find_veer(U_lower, V_lower, U_upper, V_upper):
    '''
    Find change in wind direction using U and V
    '''
    # Find wind direction at both heights
    dir_lower = mpcalc.wind_direction(U_lower.values*units('m/s'), V_lower.values*units('m/s'))
    dir_upper = mpcalc.wind_direction(U_upper.values*units('m/s'), V_upper.values*units('m/s'))
    
    # Find veer
    veer = dir_upper - dir_lower
    
    # Check for 0 wraparound issues
    if veer <= -180:
        veer += 360
    elif veer >= 180:
        veer -= 360
    
    return veer.magnitude

def find_classification(max_wspd, shear):
    '''
    Find the LLJ classification based on these values.
    
    From Vanderwende: 
            Speed    Shear
    LLJ-0:   10        5
    LLJ-1:   12        6
    LLJ-2:   16        8
    LLJ-3:   20       10
    '''
    
    if ((max_wspd>=20) and (shear>=10)):
        return 3
    elif ((max_wspd>=16) and (shear>=8)):
        return 2
    elif ((max_wspd>=12) and (shear>=6)):
        return 1
    else:
        return 0
    
def isLLJ(wspd_data, heights, U, V):
    '''
    Find out if a certain timestamp is an LLJ
    
    Parameters
    ----------
    wspd_data : wind speeds at a certain time for heights up to 2000m
    
    height_data : heights at a certain time
    
    U : u wind component at certain time
    
    V : v wind component at certain time
    
    Returns
    -------
    If there is an LLJ (True, nose_wspd, nose_height, nose_min_shear, classification, dz, veer)
    If there is not an LLJ (False, Nan, Nan, Nan, Nan, Nan, Nan)
    '''
    # define thresholds
    max_wspd_thresh = 10
    shear_thresh = 5
    
    # Select given time
    wspd = wspd_data.values
    
    # Find max wspd value
    max_wspd = np.max(wspd)
    if max_wspd < max_wspd_thresh:
        return (False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # Find nose height
    height_idx = list(wspd).index(max_wspd)
    nose_height = heights[height_idx]
    if nose_height > 1000:
        return (False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # Find min wspd above nose
    wspds_above_nose = wspd_data.sel(bottom_top=slice(height_idx, 31)).values
    min_wspd_above_nose = wspds_above_nose.min()
    
    # Find height of min wspd
    min_height_idx = list(wspd).index(min_wspd_above_nose)
    min_wspd_height = heights[min_height_idx]
    
    # Find dz
    dz = min_wspd_height - nose_height
    
    # Calculate shear
    shear = max_wspd - min_wspd_above_nose
    if shear < shear_thresh:
        return (False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    # Find the LLJ classification
    classification = find_classification(max_wspd, shear)
    
    # Find veer
    veer = find_veer(U.sel(bottom_top=height_idx), # lower U
                     V.sel(bottom_top=height_idx), # lower V
                     U.sel(bottom_top=min_height_idx), # upper U
                     V.sel(bottom_top=min_height_idx)) # upper V
    
    # If we make it through all of these calculations, there is an LLJ!
    return (True, max_wspd, nose_height, shear, classification, dz, int(veer))
