import numpy as np

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
    if hgt > 1000:
        return False
    if shear < 2.5:
        return False
    else:
        return True
    
def find_classification(ws, shear):
    '''
            max ws    shear above nose
    LLJ-0:    10          2.5
    LLJ-1     12          3
    LLJ-2     16          4
    LLJ-3     20          5
    '''
    
    if ((ws>=20) and (shear>=5)):
        return 3
    elif ((ws>=16) and (shear>=4)):
        return 2
    elif ((ws>=12) and (shear>=3)):
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


def obukhov(ust, hfx, T300):
    g = 9.8
    k = 0.4
    
    t1 = (ust**3) * T300
    t2 = k*g*hfx
    
    return (-1 * (t2/t1))


def BVF(theta_v_arr, dz, nose_idx, sfc_idx=0):
    g = 9.8
    tv = 300
    const = g / tv
    
    dtv = np.abs(theta_v_arr[nose_idx] - theta_v_arr[sfc_idx])
    
    return np.sqrt(const * (dtv / dz))


def find_veer(wd, u, l):
    veer = wd[u] - wd[l]
    
    if veer <= -180:
        veer += 360
    elif veer >= 180:
        veer -= 360
        
    return veer