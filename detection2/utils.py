import numpy as np
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

def is_LLJ(ws, hgt, shear):
    '''determine if the timestep is an LLJ'''
    if ws < 10:
        return False
    if hgt > 750:
        return False
    if shear < 3:
        return False
    else:
        return True
    
def find_classification(ws, shear):
    '''
            max ws    shear above nose
    LLJ-0:    10          3
    LLJ-1     12          5
    LLJ-2     16          8
    LLJ-3     20          10
    '''
    
    if ((ws>=20) and (shear>=10)):
        return 3
    elif ((ws>=16) and (shear>=8)):
        return 2
    elif ((ws>=12) and (shear>=5)):
        return 1
    else:
        return 0


def find_dz(z, lower, upper):
    '''
    find change in height between two indices
    
    z - array of heights
    lower - index of lower height
    upper - index of upper height
    '''
#     print(f'Lower: {lower}, upper: {upper}')
#     print(f'lower: {z[lower]}, Upper: {z[upper]}')
#     print(f'dz: {z[upper] - z[lower]}')
    
    return z[upper] - z[lower]


def obukhov(UST, T, HFX_d, r, PRESS):
    '''
    Input: timeseries of variables at the POI
    Output: The Obukhov length at the POI
    '''
    ## set up constants
    k  = 0.4
    g  = 9.81*units('m/s^2')
    Rd = 287*units('J/kg')/(1*units('K'))
    Cp = 1004.67*units('J/kg')/(1*units('K'))
    rho_Cp = 1.216e3*units('W/m^2')/(1*units('kelvin meter/second'))
    
    ### NUMERATOR
    # Calculate potential temperature using 2-m temperature and surface pressure
    TH = T*((1000*units('mbar')/PRESS)**(0.286))
    # convert to virtual potential temperature to account for moisture
    Tv = TH*(1+(0.61*r))
    numer = -1*(UST**3)*Tv
    
    ### DENOMINATOR
    # convert dynamic heat flux to kinematic heat flux
    HFX_k = HFX_d/rho_Cp
    denom = k*g*HFX_k
    
    ### calculate Obukhov length
    rmol = numer/denom
    return(rmol)


# def BVF(TH, dz, nose_idx, sfc_idx=0):
#     g = 9.8*units('m/s^2')
#     tv = 300*units('K')
#     const = g / tv
    
#     dth = TH[nose_idx] - TH[sfc_idx] # change in potential temperature with height
    
#     if dth < 0:
#         return np.nan
#     else:
#         return np.sqrt(const * (dth / dz))


def find_veer(wd, u, l):
    veer = wd[u] - wd[l]
    
    if veer <= -180:
        veer += 360
    elif veer >= 180:
        veer -= 360
        
    return veer