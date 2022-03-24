# encoding: utf-8

import time
from datetime import datetime
import numpy as np
import cartopy.crs as ccrs

import pyroms


def get_coast_from_coastpolygons(coastpolygons):
    coast = np.array([[], []]).T
    coast_nan = np.array([[np.nan], [np.nan]]).T

    for cpi in coastpolygons:
        coast = np.append(coast, np.array(cpi).T, axis=0)
        coast = np.append(coast, coast_nan, axis=0)
    return coast


def get_ijcoast(coast, grd):
    if isinstance(grd, pyroms.grid.ROMSGrid):
        x_vert = grd.hgrid.x_vert
        y_vert = grd.hgrid.y_vert
        x_rho = grd.hgrid.x_rho
        y_rho = grd.hgrid.y_rho
    elif isinstance(grd, pyroms.hgrid.CGridGeo):
        x_vert = grd.x_vert
        y_vert = grd.y_vert
        x_rho = grd.x_rho
        y_rho = grd.y_rho

    iN, jN = x_rho.shape
    ijcoast = []

    for k in range(coast.shape[0]):
        if np.isnan(coast[k, 0]):
            ijcoast.append([np.nan, np.nan])
        else:
            dist = np.abs(x_rho-coast[k, 0])+np.abs(y_rho-coast[k, 1])
            iind, jind = np.argwhere(dist == dist.min())[0]
            if (iind > 0) and (iind < iN-1) and (jind > 0) and (jind < jN-1):
                ivec = np.array([x_vert[iind+1, jind]-x_vert[iind, jind],
                                 y_vert[iind+1, jind]-y_vert[iind, jind]])
                jvec = np.array([x_vert[iind, jind+1]-x_vert[iind, jind],
                                 y_vert[iind, jind+1]-y_vert[iind, jind]])
                c = np.array([coast[k, 0]-x_vert[iind, jind],
                              coast[k, 1]-y_vert[iind, jind]])
                ifrac = np.dot(ivec, c)/(np.dot(ivec, ivec))
                jfrac = np.dot(jvec, c)/(np.dot(jvec, jvec))
                ijcoast.append([jind+jfrac, iind+ifrac])
            else:
                ijcoast.append([np.nan, np.nan])
    return np.asarray(ijcoast)


def get_grid_proj(grd, grd_type='merc', resolution='h', **kwargs):
    """
    map = get_grid_proj(grd)

    optional arguments:
      - grd_type       set projection type (default is merc)
      - resolution     set resolution parameter (default is high)

    return a Basemap object that can be use for plotting
    """

    if isinstance(grd, pyroms.grid.ROMSGrid):
        hgrid = grd.hgrid
    elif isinstance(grd, pyroms.hgrid.CGridGeo):
        hgrid = grd

    lon0, lat0 = hgrid.proj(hgrid.x_rho.mean(), hgrid.y_rho.mean(),
                            inverse=True)
    mproj = ccrs.Stereographic(central_longitude=lon0, central_latitude=lat0,
                               false_easting=0.0, false_northing=0.0)
    return mproj


def get_nc_var(varname, filename):
    """
    var = roms_nc_var(varname, filename)

    a simple wraper for netCDF4
    """

    data = pyroms.io.Dataset(filename)
    var = data.variables[varname]
    return var


def roms_varlist(option):
    """
    varlist = roms_varlist(option)

    Return ROMS varlist.
    """

    if option == 'physics':
        varlist = (['temp', 'salt', 'u', 'v', 'ubar', 'vbar', 'zeta'])
    elif option == 'physics2d':
        varlist = (['ubar', 'vbar', 'zeta'])
    elif option == 'physics3d':
        varlist = (['temp', 'salt', 'u', 'v'])
    elif option == 'mixing3d':
        varlist = (['AKv', 'AKt', 'AKs'])
    elif option == 's-param':
        varlist = (['theta_s', 'theta_b', 'Tcline', 'hc'])
    elif option == 's-coord':
        varlist = (['s_rho', 's_w', 'Cs_r', 'Cs_w'])
    elif option == 'coord':
        varlist = (['lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v'])
    elif option == 'grid':
        varlist = (['h', 'f', 'pm', 'pn', 'angle', 'lon_rho', 'lat_rho',
                    'lon_u', 'lat_u', 'lon_v', 'lat_v', 'lon_psi', 'lat_psi',
                    'mask_rho', 'mask_u', 'mask_v', 'mask_psi'])
    elif option == 'hgrid':
        varlist = (['f', 'dx', 'dy', 'angle_rho', 'lon_rho', 'lat_rho',
                    'lon_u', 'lat_u', 'lon_v', 'lat_v', 'lon_psi', 'lat_psi',
                    'mask_rho', 'mask_u', 'mask_v', 'mask_psi'])
    elif option == 'vgrid':
        varlist = (['h', 's_rho', 's_w', 'Cs_r', 'Cs_w',
                    'theta_s', 'theta_b', 'Tcline', 'hc'])
    else:
        raise Warning('Unknow varlist id')
    return varlist


def move2grid(varin, init_grid, final_grid):
    """
    tempu = move2grid(temp, 'rho', 'u')

    Move var from init_grid to final_grid.
    """

    ndim = len(varin.shape)

    if ndim == 2:

        if (init_grid == 'rho' and final_grid == 'u'):
            varout = 0.5 * (varin[:, 1:] + varin[:, :-1])
        elif (init_grid == 'rho' and final_grid == 'v'):
            varout = 0.5 * (varin[1:, :] + varin[:-1, :])
        elif (init_grid == 'rho' and final_grid == 'psi'):
            varout = 0.25 * (varin[1:, 1:] + varin[:-1, :-1] +
                             varin[1:, :-1] + varin[:-1, 1:])
        elif (init_grid == 'u' and final_grid == 'psi'):
            varout = 0.5 * (varin[1:, :] + varin[:-1, :])
        elif (init_grid == 'v' and final_grid == 'psi'):
            varout = 0.5 * (varin[:, 1:] + varin[:, :-1])
        else:
            raise ValueError(
                'Undefined combination for init_grid and final_grid')

    elif ndim == 3:

        if (init_grid == 'rho' and final_grid == 'u'):
            varout = 0.5 * (varin[:, :, 1:] + varin[:, :, :-1])
        elif (init_grid == 'rho' and final_grid == 'v'):
            varout = 0.5 * (varin[:, 1:, :] + varin[:, :-1, :])
        elif (init_grid == 'rho' and final_grid == 'psi'):
            varout = 0.25 * (varin[:, 1:, 1:] + varin[:, :-1, :-1] +
                             varin[:, 1:, :-1] + varin[:, :-1, 1:])
        elif (init_grid == 'u' and final_grid == 'psi'):
            varout = 0.5 * (varin[:, 1:, :] + varin[:, :-1, :])
        elif (init_grid == 'v' and final_grid == 'psi'):
            varout = 0.5 * (varin[:, :, 1:] + varin[:, :, :-1])
        else:
            raise ValueError(
                'Undefined combination for init_grid and final_grid')

    else:
        raise ValueError('varin must be 2D or 3D')
    return varout


def get_date_tag(roms_time, ref=(2006, 0o1, 0o1),
                 format="%d %b %Y at %H:%M:%S"):
    """
    tag = get_date_tag(roms_time)

    return date tag for roms_time (in second since initialisation).
    default reference time is January 1st 2006.
    """

    ref = time.mktime(datetime(ref[0], ref[1], ref[2]).timetuple())
    timestamp = ref + roms_time
    tag = datetime.fromtimestamp(timestamp).strftime(format)
    return tag


def apply_mask_change(file, grd):
    """
    Apply mask change saved by edit_mesh_mask in the mask_change.txt file
    """

    mask_changes = open(file, 'r')
    lines = mask_changes.readlines()
    mask_changes.close()

    for line in lines:
        s = line.split()
        i = int(s[0])
        j = int(s[1])
        mask = float(s[2])
        grd.hgrid.mask_rho[j, i] = mask
    return


def get_lonlat(iindex, jindex, grd, Cpos='rho'):
    """
    lon, lat = get_lonlat(iindex, jindex, grd)

    return the longitude (degree east) and latitude (degree north)
    for grid point (iindex, jindex)
    """

    if Cpos == 'u':
        lon = grd.hgrid.lon_u[:, :]
        lat = grd.hgrid.lat_u[:, :]
    elif Cpos == 'v':
        lon = grd.hgrid.lon_v[:, :]
        lat = grd.hgrid.lat_v[:, :]
    elif Cpos == 'rho':
        lon = grd.hgrid.lon_rho[:, :]
        lat = grd.hgrid.lat_rho[:, :]
    elif Cpos == 'psi':
        lon = grd.hgrid.lon_psi[:, :]
        lat = grd.hgrid.lat_psi[:, :]
    else:
        raise Warning('%s bad position. Cpos must be rho, psi, u or v.' % Cpos)
    return lon[jindex, iindex], lat[jindex, iindex]


def get_ij(longitude, latitude, grd, Cpos='rho'):
    """
    i, j = get_ij(longitude, latitude, grd)

    return the index of the closest point on the grid from the
    point (longitude,latitude) in degree
    """

    if Cpos == 'u':
        lon = grd.hgrid.lon_u[:, :]
        lat = grd.hgrid.lat_u[:, :]
    elif Cpos == 'v':
        lon = grd.hgrid.lon_v[:, :]
        lat = grd.hgrid.lat_v[:, :]
    elif Cpos == 'rho':
        lon = grd.hgrid.lon_rho[:, :]
        lat = grd.hgrid.lat_rho[:, :]
    elif Cpos == 'psi':
        lon = grd.hgrid.lon_psi[:, :]
        lat = grd.hgrid.lat_psi[:, :]
    else:
        raise Warning('%s bad position. Cpos must be rho, psi, u or v.' % Cpos)

    lon = lon[:, :] - longitude
    lat = lat[:, :] - latitude

    diff = (lon * lon) + (lat * lat)

    jindex, iindex = np.where(diff == diff.min())
    return iindex[0], jindex[0]


def get_coast_from_map(map):
    coast = []
    kk = len(map.coastsegs)
    for k in range(kk):
        ll = len(map.coastsegs[k])
        for li in range(ll):
            c = list(map(
                map.coastsegs[k][li][0],
                map.coastsegs[k][li][1], inverse=True))
            coast.append(c)
        coast.append((np.nan, np.nan))
    return np.asarray(coast)
