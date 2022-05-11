"""
Regridding utilities.
"""

from typing import List, Tuple, Union

import numpy as np
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
import netCDF4 as nc
import xarray as xr
import xesmf as xe
import pyroms


_fill_value = 1.e10
_vtype = Union[type(None), List, Tuple]


def make_ini(dsi: xr.DataArray,
             grd: pyroms.grid.ROMSGrid,
             filename: str = 'roms_ini.nc',
             t: str = 'ocean_time', z: str = 'z',
             x: str = 'lon', y: str = 'lat',
             flood: bool = True, flood_raw: bool = True,
             sigma: float = 3.,
             vars_in: _vtype = None, vars_out: _vtype = None):
    """
    Construct ROMS initial file from z-coordinate model/reanalyasis products.

    usage:

    dsr, dsi, dso = make_ini(dsi: xr.DataArray,
                             grd: pyroms.grid.ROMSGrid,
                             filename: str = 'roms_ini.nc',
                             t: str = 'ocean_time', z: str = 'z',
                             x: str = 'lon', y: str = 'lat',
                             flood: bool = True, flood_raw: bool = True,
                             sigma: float = 3.,
                             vars_in: _vtype = None, vars_out: _vtype = None):

    inputs:
        dsi        - Xarray dataset containing the raw data
        grd        - pyroms grid object of regridding grid
        filename   - name of outputting initial file
        t, z, x, y - string, name of each dimesion (time, depth, lon, lat) in
                     the inputting dataset dsi
        flood      - boolean, if flood the regridded data
        flood_raw  - boolean, if flood the raw data
        sigma      - float, when flood the grid, sigma is used as a parameter
                     in the Gaussian filter (scipy.ndimage.gaussian_filter).
        vars_in    - names of variables in input DataArrays
        vars_out   - names of variables in output DataArrays
    outputs:
        dsr        - Xarray dataset of regridded data
        dsi        - dict, containing Xarray dataset of input data
        dso        - dict, containing Xarray dataset of output data
    """

    # Data preprocessing and interpolation
    dsi, dso = data_prep(dsi, grd, t=t, z=z, x=x, y=y, flood_raw=flood_raw,
                         vars_in=vars_in, vars_out=vars_out)
    dsr = interp(dsi, dso, flood=flood, flood_raw=flood_raw, sigma=sigma)

    # Rotate velocities if provided in the dataset
    if 'ue' in dsr and 'vn' in dsr:
        vel = (dsr.ue + 1j*dsr.vn) * \
              (np.cos(dso.angle) - 1j*np.sin(dso.angle))
        dsr['u'], dsr['v'] = np.real(vel), np.imag(vel)
        dsr.drop_vars(['ue', 'vn'])
    if 'uicee' in dsr and 'vicen' in dsr:
        vel = (dsr.uicee + 1j*dsr.vicen) * \
              (np.cos(dso.angle) - 1j*np.sin(dso.angle))
        dsr['uice'], dsr['vice'] = np.real(vel), np.imag(vel)
        dsr.drop_vars(['uicee', 'vicen'])

    # Mask invalid values
    dsr = dsr.where(dso.mask_rho == 1, other=_fill_value)
    if dsr.isnull().any():
        print('\nNaN values in regridded data! Convert to 0.\n')
        dsr = dsr.where(~dsr.isnull(), 0.)

    # Generate masks for u/v points
    mask_u = dso.mask_rho.interp(xi_rho=dso.xi_rho[:-1]+0.5)
    mask_v = dso.mask_rho.interp(eta_rho=dso.eta_rho[:-1]+0.5)
    mask_u = mask_u.where(mask_u == 1, 0)
    mask_v = mask_v.where(mask_v == 1, 0)

    # Write the regridded data to a ROMS file
    pyroms.io.nc_create_roms_file(filename, grd)

    fh = nc.Dataset(filename, 'a')
    fh.variables['ocean_time'][:] = 0
    if 'units' in dsi.ocean_time.attrs:
        fh.variables['ocean_time'].units = dsi.ocean_time.attrs['units']

    if vars_in is not None:
        if 'salt' in vars_out:
            pyroms.io.nc_write_var(
                fh, dsr.salt, 'salt',
                ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
                long_name='salinity',
                fill_value=_fill_value)
        if 'temp' in vars_out:
            pyroms.io.nc_write_var(
                fh, dsr.temp, 'temp',
                ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
                long_name='temperature', units='Celsius',
                fill_value=_fill_value)
        if 'zeta' in vars_out:
            pyroms.io.nc_write_var(
                fh, dsr.zeta, 'zeta',
                ('ocean_time', 'eta_rho', 'xi_rho'),
                long_name='free-surface', units='meter',
                fill_value=_fill_value)
        if 'u' in vars_out:
            urho = dsr.u
            u2rho = (urho*dso.dz).sum(dim='s_rho')/dso.h
            u = urho.interp(xi_rho=urho.xi_rho[:-1]+0.5)
            u2 = u2rho.interp(xi_rho=u2rho.xi_rho[:-1]+0.5)
            u = u.where(mask_u == 1, other=_fill_value)
            u2 = u2.where(mask_u == 1, other=_fill_value)
            pyroms.io.nc_write_var(
                fh, u, 'u',
                ('ocean_time', 's_rho', 'eta_u', 'xi_u'),
                long_name='u-momentum component',
                units='meter second-1',
                fill_value=_fill_value)
            pyroms.io.nc_write_var(
                fh, u2, 'ubar',
                ('ocean_time', 'eta_u', 'xi_u'),
                long_name='vertically integrated u-momentum component',
                units='meter second-1',
                fill_value=_fill_value)
        if 'v' in vars_out:
            vrho = dsr.v
            v2rho = (vrho*dso.dz).sum(dim='s_rho')/dso.h
            v = vrho.interp(eta_rho=vrho.eta_rho[:-1]+0.5)
            v2 = v2rho.interp(eta_rho=v2rho.eta_rho[:-1]+0.5)
            v = v.where(mask_v == 1, other=_fill_value)
            v2 = v2.where(mask_v == 1, other=_fill_value)
            pyroms.io.nc_write_var(
                fh, v, 'v',
                ('ocean_time', 's_rho', 'eta_v', 'xi_v'),
                long_name='v-momentum component',
                units='meter second-1',
                fill_value=_fill_value)
            pyroms.io.nc_write_var(
                fh, v2, 'vbar',
                ('ocean_time', 'eta_v', 'xi_v'),
                long_name='vertically integrated v-momentum component',
                units='meter second-1',
                fill_value=_fill_value)
        if 'aice' in vars_out:
            pyroms.io.nc_write_var(
                fh, dsr.aice, 'aice',
                ('ocean_time', 'eta_rho', 'xi_rho'),
                long_name='sea ice concentration',
                fill_value=_fill_value)
        if 'hice' in vars_out:
            pyroms.io.nc_write_var(
                fh, dsr.hice, 'hice',
                ('ocean_time', 'eta_rho', 'xi_rho'),
                long_name='sea ice thickness',
                units='meter',
                fill_value=_fill_value)
        if 'uice' in vars_out:
            uice = dsr.uice.interp(xi_rho=dsr.uice.xi_rho[:-1]+0.5)
            uice = uice.where(mask_u == 1, other=_fill_value)
            pyroms.io.nc_write_var(
                fh, uice, 'uice',
                ('ocean_time', 'eta_u', 'xi_u'),
                long_name='sea ice velocity u-component',
                units='meter second-1',
                fill_value=_fill_value)
        if 'vice' in vars_out:
            vice = dsr.vice.interp(eta_rho=dsr.uice.eta_rho[:-1]+0.5)
            vice = vice.where(mask_v == 1, other=_fill_value)
            pyroms.io.nc_write_var(
                fh, vice, 'vice',
                ('ocean_time', 'eta_v', 'xi_v'),
                long_name='sea ice velocity v-component',
                units='meter second-1',
                fill_value=_fill_value)
    else:
        if 'salt' in dsr:
            pyroms.io.nc_write_var(
                fh, dsr.salt, 'salt',
                ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
                long_name='salinity',
                fill_value=_fill_value)
        if 'temp' in dsr:
            pyroms.io.nc_write_var(
                fh, dsr.temp, 'temp',
                ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
                long_name='temperature', units='Celsius',
                fill_value=_fill_value)
        if 'zeta' in dsr:
            pyroms.io.nc_write_var(
                fh, dsr.zeta, 'zeta',
                ('ocean_time', 'eta_rho', 'xi_rho'),
                long_name='free-surface', units='meter',
                fill_value=_fill_value)
        else:
            pyroms.io.nc_write_var(
                fh, 0, 'zeta',
                ('ocean_time', 'eta_rho', 'xi_rho'),
                long_name='free-surface', units='meter',
                fill_value=_fill_value)
        if 'u' in dsr:
            urho = dsr.u
            u2rho = (urho*dso.dz).sum(dim='s_rho')/dso.h
            u = urho.interp(xi_rho=urho.xi_rho[:-1]+0.5)
            u2 = u2rho.interp(xi_rho=u2rho.xi_rho[:-1]+0.5)
            u = u.where(mask_u == 1, other=_fill_value)
            u2 = u2.where(mask_u == 1, other=_fill_value)
        else:
            u, u2 = 0., 0.
        pyroms.io.nc_write_var(
            fh, u, 'u',
            ('ocean_time', 's_rho', 'eta_u', 'xi_u'),
            long_name='u-momentum component',
            units='meter second-1',
            fill_value=_fill_value)
        pyroms.io.nc_write_var(
            fh, u2, 'ubar',
            ('ocean_time', 'eta_u', 'xi_u'),
            long_name='vertically integrated u-momentum component',
            units='meter second-1',
            fill_value=_fill_value)
        if 'v' in dsr:
            vrho = dsr.v
            v2rho = (vrho*dso.dz).sum(dim='s_rho')/dso.h
            v = vrho.interp(eta_rho=vrho.eta_rho[:-1]+0.5)
            v2 = v2rho.interp(eta_rho=v2rho.eta_rho[:-1]+0.5)
            v = v.where(mask_v == 1, other=_fill_value)
            v2 = v2.where(mask_v == 1, other=_fill_value)
        else:
            v, v2 = 0., 0.
        pyroms.io.nc_write_var(
            fh, v, 'v',
            ('ocean_time', 's_rho', 'eta_v', 'xi_v'),
            long_name='v-momentum component',
            units='meter second-1',
            fill_value=_fill_value)
        pyroms.io.nc_write_var(
            fh, v2, 'vbar',
            ('ocean_time', 'eta_v', 'xi_v'),
            long_name='vertically integrated v-momentum component',
            units='meter second-1',
            fill_value=_fill_value)

    fh.close()

    return dsr, dsi, dso


def make_bry(dsi: xr.DataArray,
             grd: pyroms.grid.ROMSGrid,
             filename: str = 'roms_bry.nc',
             t: str = 'ocean_time', z: str = 'z',
             x: str = 'lon', y: str = 'lat',
             flood: bool = True, flood_raw: bool = True,
             sigma: float = 3.,
             vars_in: _vtype = None, vars_out: _vtype = None,
             is_dir: List = [True, True, True, True]):
    """
    Construct ROMS boundary file from z-coordinate model/reanalyasis products.

    usage:
    dsr, dsi, dso = make_bry(dsi: xr.DataArray,
                             grd: pyroms.grid.ROMSGrid,
                             filename: str = 'roms_ini.nc',
                             t: str = 'ocean_time', z: str = 'z',
                             x: str = 'lon', y: str = 'lat',
                             flood: bool = True, flood_raw: bool = True,
                             sigma: float = 3.,
                             vars_in: _vtype = None, vars_out: _vtype = None,
                             is_dir = [True, True, True, True]):

    inputs:
        dsi        - Xarray dataset containing the raw data
        grd        - pyroms grid object of regridding grid
        filename   - name of outputting boundary file
        t, z, x, y - string, name of each dimesion (time, depth, lon, lat) in
                     the inputting dataset dsi
        flood      - boolean, if flood the regridded data
        flood_raw  - boolean, if flood the raw data
        sigma      - float, when flood the grid, sigma is used as a parameter
                     in the Gaussian filter (scipy.ndimage.gaussian_filter).
        vars_in    - names of variables in input DataArrays
        vars_out   - names of variables in output DataArrays
        is_dir     - boolean of list[4], whether to process for each boundary,
                     ordered [west, east, south, north]. Default to process
                     all boundaries.
    outputs:
        dsr_bry    - dict, containing Xarray dataset of regridded boundary data
        dsi_bry    - dict, containing Xarray dataset of input data at boundary
        dso_bry    - dict, containing Xarray dataset of output data at boundary
    """

    # Data preprocessing.
    dsi, dso = data_prep(dsi, grd, t=t, z=z, x=x, y=y, flood_raw=flood_raw,
                         vars_in=vars_in, vars_out=vars_out)

    # Generate an empty boundary file before looping over boundaries.
    pyroms.io.nc_create_roms_bdry_file(filename, grd)
    fh = nc.Dataset(filename, 'a')
    fh.variables['ocean_time'][:] = dsi.ocean_time
    if 'units' in dsi.ocean_time.attrs:
        fh.variables['ocean_time'].units = dsi.ocean_time.attrs['units']

    # Generate dict to store boundary Xarrays.
    dsi_bry = {}
    dso_bry = {}
    dsr_bry = {}

    # Loop over boundaries to perform regridding.
    for i, var_dir in enumerate(['west', 'east', 'south', 'north']):
        if is_dir[i]:
            # Slice into dataset.
            if var_dir == 'west':
                dso_b = dso.isel(xi_rho=[0, 1])
                slc = dict(xi_rho=0)
            elif var_dir == 'east':
                dso_b = dso.isel(xi_rho=[-2, -1])
                slc = dict(xi_rho=1)
            elif var_dir == 'south':
                dso_b = dso.isel(eta_rho=[0, 1])
                slc = dict(eta_rho=0)
            elif var_dir == 'north':
                dso_b = dso.isel(eta_rho=[-2, -1])
                slc = dict(eta_rho=1)

            # Only use data in the boundary region
            lon, lat = np.meshgrid(dsi.lon, dsi.lat)
            x, y = grd.hgrid.proj(lon, lat)
            dsi.coords['x'], dsi.coords['y'] = \
                (['lat', 'lon'], x), (['lat', 'lon'], y)

            xmin, xmax = dso_b.x_rho.min().item(), dso_b.x_rho.max().item()
            ymin, ymax = dso_b.y_rho.min().item(), dso_b.y_rho.max().item()
            dx, dy = xmax - xmin, ymax - ymin
            xmin, xmax = xmin - 0.1*dx, xmax + 0.1*dx
            ymin, ymax = ymin - 0.1*dy, ymax + 0.1*dy

            dsi_b = dsi.where((dsi.x > xmin) & (dsi.x < xmax) &
                              (dsi.y > ymin) & (dsi.y < ymax), drop=True)

            # Drop depth slices without valid data
            if 'z' in dsi_b.dims:
                dsi_b = dsi_b.dropna(dim='z', how='all')

            # Perform interpolation
            dsr = interp(dsi_b, dso_b,
                         flood=flood, flood_raw=flood_raw, sigma=sigma)
            dsr = dsr.where(dso_b.mask_rho == 1, _fill_value)

            # Rotate velocity if provided in the dataset
            if 'ue' in dsr and 'vn' in dsr:
                vel = (dsr.ue + 1j*dsr.vn) * \
                      (np.cos(dso.angle) - 1j*np.sin(dso.angle))
                dsr['u'], dsr['v'] = np.real(vel), np.imag(vel)
            if 'uicee' in dsr and 'vicen' in dsr:
                vel = (dsr.uicee + 1j*dsr.vicen) * \
                      (np.cos(dso.angle) - 1j*np.sin(dso.angle))
                dsr['uice'], dsr['vice'] = np.real(vel), np.imag(vel)

            dsr.coords['zr'] = dso_b.zr

            # Attach xarray data pieces to dict
            dsi_bry[var_dir] = dsi_b
            dso_bry[var_dir] = dso_b
            dsr_bry[var_dir] = dsr

            # Mask invalid values
            dsr = dsr.where(dso_b.mask_rho == 1, other=_fill_value)
            if dsr.isnull().any():
                print('\nNaN values in regridded data! Convert to 0.\n')
                dsr = dsr.where(~dsr.isnull(), 0.)

            # Generate masks for u/v points
            if var_dir == 'west':
                mask_u = dso_b.mask_rho.mean(dim='xi_rho')
                mask_v = dso_b.mask_rho.interp(eta_rho=dso_b.eta_rho[:-1]+0.5,
                                               xi_rho=dso_b.xi_rho[0])
            elif var_dir == 'east':
                mask_u = dso_b.mask_rho.mean(dim='xi_rho')
                mask_v = dso_b.mask_rho.interp(eta_rho=dso_b.eta_rho[:-1]+0.5,
                                               xi_rho=dso_b.xi_rho[1])
            elif var_dir == 'south':
                mask_u = dso_b.mask_rho.interp(xi_rho=dso_b.xi_rho[:-1]+0.5,
                                               eta_rho=dso_b.eta_rho[0])
                mask_v = dso_b.mask_rho.mean(dim='eta_rho')
            elif var_dir == 'north':
                mask_u = dso_b.mask_rho.interp(xi_rho=dso_b.xi_rho[:-1]+0.5,
                                               eta_rho=dso_b.eta_rho[1])
                mask_v = dso_b.mask_rho.mean(dim='eta_rho')
            mask_u = mask_u.where(mask_u == 1, 0)
            mask_v = mask_v.where(mask_v == 1, 0)

            # Last dimension of the boundary
            if var_dir in ['west', 'east']:
                dim = ('eta_rho',)
                dimu = ('eta_u',)
                dimv = ('eta_v',)
            else:
                dim = ('xi_rho',)
                dimu = ('xi_u',)
                dimv = ('xi_v',)

            # Write to a ROMS file
            if vars_in is not None:
                if 'salt' in vars_out:
                    pyroms.io.nc_write_var(
                        fh, dsr.salt.isel(slc), 'salt_' + var_dir,
                        ('ocean_time', 's_rho') + dim,
                        long_name='salinity at ' + var_dir + ' boundary',
                        fill_value=_fill_value)
                if 'temp' in vars_out:
                    pyroms.io.nc_write_var(
                        fh, dsr.temp.isel(slc), 'temp_' + var_dir,
                        ('ocean_time', 's_rho') + dim,
                        long_name='temperature at ' + var_dir + ' boundary',
                        units='Celsius',
                        fill_value=_fill_value)
                if 'zeta' in vars_out:
                    pyroms.io.nc_write_var(
                        fh, dsr.zeta.isel(slc), 'zeta_' + var_dir,
                        ('ocean_time',) + dim,
                        long_name='free-surface at ' + var_dir + ' boundary',
                        units='meter',
                        fill_value=_fill_value)
                if 'u' in vars_out:
                    urho = dsr.u
                    u2rho = (urho*dso_b.dz).sum(dim='s_rho')/dso_b.h
                    if var_dir in ['west', 'east']:
                        u = urho.mean(dim='xi_rho')
                        u2 = u2rho.mean(dim='xi_rho')
                    if var_dir == 'south':
                        u = urho.interp(eta_rho=urho.eta_rho[0],
                                        xi_rho=urho.xi_rho[:-1]+0.5)
                        u2 = u2rho.interp(eta_rho=u2rho.eta_rho[0],
                                          xi_rho=u2rho.xi_rho[:-1]+0.5)
                    if var_dir == 'north':
                        u = urho.interp(eta_rho=urho.eta_rho[1],
                                        xi_rho=urho.xi_rho[:-1]+0.5)
                        u2 = u2rho.interp(eta_rho=u2rho.eta_rho[1],
                                          xi_rho=u2rho.xi_rho[:-1]+0.5)
                    u = u.where(mask_u == 1, 0)
                    u2 = u2.where(mask_u == 1, 0)
                    pyroms.io.nc_write_var(
                        fh, u, 'u_' + var_dir,
                        ('ocean_time', 's_rho') + dimu,
                        long_name='u-momentum component at ' + var_dir +
                                  ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
                    pyroms.io.nc_write_var(
                        fh, u2, 'ubar_' + var_dir,
                        ('ocean_time',) + dimu,
                        long_name='vertically integrated ' +
                                  'u-momentum component at ' + var_dir +
                                  ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
                if 'v' in vars_out:
                    vrho = dsr.v
                    v2rho = (vrho*dso_b.dz).sum(dim='s_rho')/dso_b.h
                    if var_dir == 'west':
                        v = vrho.interp(eta_rho=vrho.eta_rho[:-1]+0.5,
                                        xi_rho=vrho.xi_rho[0])
                        v2 = v2rho.interp(eta_rho=v2rho.eta_rho[:-1]+0.5,
                                          xi_rho=v2rho.xi_rho[0])
                    if var_dir == 'east':
                        v = vrho.interp(eta_rho=vrho.eta_rho[:-1]+0.5,
                                        xi_rho=vrho.xi_rho[1])
                        v2 = v2rho.interp(eta_rho=v2rho.eta_rho[:-1]+0.5,
                                          xi_rho=v2rho.xi_rho[1])
                    if var_dir in ['south', 'north']:
                        v = vrho.mean(dim='eta_rho')
                        v2 = v2rho.mean(dim='eta_rho')
                    v = v.where(mask_v == 1, 0)
                    v2 = v2.where(mask_v == 1, 0)
                    pyroms.io.nc_write_var(
                        fh, v, 'v_' + var_dir,
                        ('ocean_time', 's_rho') + dimv,
                        long_name='v-momentum component at ' + var_dir +
                                  ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
                    pyroms.io.nc_write_var(
                        fh, v2, 'vbar_' + var_dir,
                        ('ocean_time',) + dimv,
                        long_name='vertically integrated ' +
                                  'v-momentum component at ' + var_dir +
                                  ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
                if 'aice' in vars_out:
                    pyroms.io.nc_write_var(
                        fh, dsr.aice.isel(slc), 'aice_' + var_dir,
                        ('ocean_time',) + dim,
                        long_name='sea ice concentration at ' +
                                  var_dir + ' boundary',
                        fill_value=_fill_value)
                if 'hice' in vars_out:
                    pyroms.io.nc_write_var(
                        fh, dsr.hice.isel(slc), 'hice_' + var_dir,
                        ('ocean_time',) + dim,
                        long_name='sea ice thickness ' + var_dir +
                                  ' boundary',
                        units='meter',
                        fill_value=_fill_value)
                if 'uice' in vars_out:
                    urho = dsr.uice
                    if var_dir in ['west', 'east']:
                        u = urho.mean(dim='xi_rho')
                    if var_dir == 'south':
                        u = urho.interp(eta_rho=urho.eta_rho[0],
                                        xi_rho=urho.xi_rho[:-1]+0.5)
                    if var_dir == 'north':
                        u = urho.interp(eta_rho=urho.eta_rho[1],
                                        xi_rho=urho.xi_rho[:-1]+0.5)
                    u = u.where(mask_u == 1, 0)
                    pyroms.io.nc_write_var(
                        fh, u, 'uice_' + var_dir,
                        ('ocean_time',) + dimu,
                        long_name='sea ice velocity u-component at ' +
                                  var_dir + ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
                if 'vice' in vars_out:
                    vrho = dsr.vice
                    if var_dir == 'west':
                        v = vrho.interp(eta_rho=vrho.eta_rho[:-1]+0.5,
                                        xi_rho=vrho.xi_rho[0])
                    if var_dir == 'east':
                        v = vrho.interp(eta_rho=vrho.eta_rho[:-1]+0.5,
                                        xi_rho=vrho.xi_rho[1])
                    if var_dir in ['south', 'north']:
                        v = vrho.mean(dim='eta_rho')
                    v = v.where(mask_v == 1, 0)
                    pyroms.io.nc_write_var(
                        fh, v, 'vice_' + var_dir,
                        ('ocean_time',) + dimv,
                        long_name='sea ice velocity v-component at ' +
                                  var_dir + ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
            else:
                if 'salt' in dsr:
                    pyroms.io.nc_write_var(
                        fh, dsr.salt.isel(slc), 'salt_' + var_dir,
                        ('ocean_time', 's_rho') + dim,
                        long_name='salinity at ' + var_dir + ' boundary',
                        fill_value=_fill_value)
                if 'temp' in dsr:
                    pyroms.io.nc_write_var(
                        fh, dsr.temp.isel(slc), 'temp_' + var_dir,
                        ('ocean_time', 's_rho') + dim,
                        long_name='temperature at ' + var_dir + ' boundary',
                        units='Celsius',
                        fill_value=_fill_value)
                if 'zeta' in dsr:
                    pyroms.io.nc_write_var(
                        fh, dsr.zeta.isel(slc), 'zeta_' + var_dir,
                        ('ocean_time',) + dim,
                        long_name='free-surface at ' + var_dir + ' boundary',
                        units='meter',
                        fill_value=_fill_value)
                if 'u' in dsr:
                    urho = dsr.u
                    u2rho = (urho*dso_b.dz).sum(dim='s_rho')/dso_b.h
                    if var_dir in ['west', 'east']:
                        u = urho.mean(dim='xi_rho')
                        u2 = u2rho.mean(dim='xi_rho')
                    if var_dir == 'south':
                        u = urho.interp(eta_rho=urho.eta_rho[0],
                                        xi_rho=urho.xi_rho[:-1]+0.5)
                        u2 = u2rho.interp(eta_rho=u2rho.eta_rho[0],
                                          xi_rho=u2rho.xi_rho[:-1]+0.5)
                    if var_dir == 'north':
                        u = urho.interp(eta_rho=urho.eta_rho[1],
                                        xi_rho=urho.xi_rho[:-1]+0.5)
                        u2 = u2rho.interp(eta_rho=u2rho.eta_rho[1],
                                          xi_rho=u2rho.xi_rho[:-1]+0.5)
                    u = u.where(mask_u == 1, 0)
                    u2 = u2.where(mask_u == 1, 0)
                    pyroms.io.nc_write_var(
                        fh, u, 'u_' + var_dir,
                        ('ocean_time', 's_rho') + dimu,
                        long_name='u-momentum component at ' + var_dir +
                                  ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
                    pyroms.io.nc_write_var(
                        fh, u2, 'ubar_' + var_dir,
                        ('ocean_time',) + dimu,
                        long_name='vertically integrated ' +
                                  'u-momentum component at ' + var_dir +
                                  ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
                if 'v' in dsr:
                    vrho = dsr.v
                    v2rho = (vrho*dso_b.dz).sum(dim='s_rho')/dso_b.h
                    if var_dir == 'west':
                        v = vrho.interp(eta_rho=vrho.eta_rho[:-1]+0.5,
                                        xi_rho=vrho.xi_rho[0])
                        v2 = v2rho.interp(eta_rho=v2rho.eta_rho[:-1]+0.5,
                                          xi_rho=v2rho.xi_rho[0])
                    if var_dir == 'east':
                        v = vrho.interp(eta_rho=vrho.eta_rho[:-1]+0.5,
                                        xi_rho=vrho.xi_rho[1])
                        v2 = v2rho.interp(eta_rho=v2rho.eta_rho[:-1]+0.5,
                                          xi_rho=v2rho.xi_rho[1])
                    if var_dir in ['south', 'north']:
                        v = vrho.mean(dim='eta_rho')
                        v2 = v2rho.mean(dim='eta_rho')
                    v = v.where(mask_v == 1, 0)
                    v2 = v2.where(mask_v == 1, 0)
                    pyroms.io.nc_write_var(
                        fh, v, 'v_' + var_dir,
                        ('ocean_time', 's_rho') + dimv,
                        long_name='v-momentum component at ' + var_dir +
                                  ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)
                    pyroms.io.nc_write_var(
                        fh, v2, 'vbar_' + var_dir,
                        ('ocean_time',) + dimv,
                        long_name='vertically integrated ' +
                                  'v-momentum component at ' + var_dir +
                                  ' boundary',
                        units='meter second-1',
                        fill_value=_fill_value)

    fh.close()

    return dsr_bry, dsi_bry, dso_bry


def data_prep(dsi: xr.DataArray,
              grd: pyroms.grid.ROMSGrid,
              t: str, z: str, x: str, y: str,
              flood_raw: bool = False,
              vars_in: _vtype = None, vars_out: _vtype = None,
              use_vars: _vtype = None):

    """
    Pre-process the input xarray to meet xesmf convention.

    This function rename and reorder the dimensions of the input dataset,
    and drop depth levels without any data. It alsl generate masks for the
    input dataset if flooding of raw data is performed.

    Usage:
        dsi, dso = data_prep(dsi, grd, t, z, x, y, flood_raw=True,
                             vars_in=None, vars_out=None, use_vars=None)

    Inputs:
        dsi        - Xarray dataset containing the raw data
        grd        - pyroms grid object of regridding grid
        t, z, x, y - string, name of each dimesion (time, depth, lon, lat) in
                     the inputting dataset dsi
        flood_raw  - boolean, if flood the raw data
        vars_in    - names of variables in input DataArrays
        vars_out   - names of variables in output DataArrays
        use_vars   - list of strings, if provided only take these vars in
                     the input dataset

    outputs:
        dsi        - Xarray dataset of the reformatted raw data
        dso        - Xarray dataset of the regridding grid
    """

    # Rename and transpose dataset
    if z in dsi.dims:
        dsi = dsi.rename({t: 'ocean_time', z: 'z', x: 'lon', y: 'lat'})
        dsi = dsi.transpose('ocean_time', 'z', 'lat', 'lon')
        # Drop depth slices without valid data.
        dsi = dsi.dropna(dim='z', how='all')
    else:
        dsi = dsi.rename({t: 'ocean_time', x: 'lon', y: 'lat'})
        dsi = dsi.transpose('ocean_time', 'lat', 'lon')

    # Rename variables.
    if vars_in is not None:
        assert len(vars_in) == len(vars_out), \
            'vars_in/vars_out must have the same length.'
        for vari, varo in zip(vars_in, vars_out):
            dsi = dsi.rename({vari: varo})
        dsi = dsi[vars_out]
        for uvraw, uv in zip(['u', 'v', 'uice', 'vice'],
                             ['ue', 'vn', 'uicee', 'vicen']):
            if uvraw in dsi:
                dsi.rename({uvraw: uv})
    else:
        for var in dsi.data_vars:
            long_name = None
            if 'long_name' in dsi[var].attrs:
                long_name = 'long_name'
            elif 'longname' in dsi[var].attrs:
                long_name = 'longname'

            if long_name is not None:
                if 'salinity' in dsi[var].attrs[long_name].lower():
                    dsi = dsi.rename({var: 'salt'})
                elif 'temperature' in dsi[var].attrs[long_name].lower():
                    dsi = dsi.rename({var: 'temp'})
                elif 'height' in dsi[var].attrs[long_name].lower():
                    dsi = dsi.rename({var: 'zeta'})
                elif 'velocity' in dsi[var].attrs[long_name].lower():
                    if 'east' in dsi[var].attrs[long_name].lower():
                        dsi = dsi.rename({var: 'ue'})
                    elif 'north' in dsi[var].attrs[long_name].lower():
                        dsi = dsi.rename({var: 'vn'})
        if use_vars is not None:
            dsi = dsi[use_vars]

    if flood_raw:
        # Construct grid vortex coordinates
        lon, lat = dsi.lon.data, dsi.lat.data
        if lon.ndim == 1:
            lon_b = np.zeros(len(lon)+1)
            lon_b[1:-1] = 0.5*(lon[1:]+lon[:-1])
            lon_b[0] = lon_b[1] - (lon[1]-lon[0])
            lon_b[-1] = lon_b[-2] + (lon[-1]-lon[-2])
            lat_b = np.zeros(len(lat)+1)
            lat_b[1:-1] = 0.5*(lat[1:]+lat[:-1])
            lat_b[0] = lat_b[1] - (lat[1]-lat[0])
            lat_b[-1] = lat_b[-2] + (lat[-1]-lat[-2])

            lon_b, lat_b = np.meshgrid(lon_b, lat_b)

        elif lon.ndim == 2:
            lon_p = 0.25*(lon[2:, 2:]+lon[:-2, 2:]+lon[2:, :-2]+lon[:-2, :-2])
            lat_p = 0.25*(lat[2:, 2:]+lat[:-2, 2:]+lat[2:, :-2]+lat[:-2, :-2])
            lon_b, lat_b = pyroms.rho_to_vert_geo(lon, lat, lon_p, lat_p,
                                                  grd.hgrid.proj)

        dsi.coords['lon_b'] = (['x_b', 'y_b'], lon_b)
        dsi.coords['lat_b'] = (['x_b', 'y_b'], lat_b)

    # Convert ROMS grid object to Xarray dataset and pre-process for xesmf.
    dso = grd.to_xarray()
    if flood_raw:
        dso = dso[['lon_rho', 'lat_rho', 'lon_vert', 'lat_vert', 's_rho',
                   'mask_rho', 'angle', 'h']]
        dso = dso.rename({'lon_rho': 'lon', 'lat_rho': 'lat',
                          'lon_vert': 'lon_b', 'lat_vert': 'lat_b'})
    else:
        dso = dso[['lon_rho', 'lat_rho', 's_rho', 'mask_rho', 'angle', 'h']]
        dso = dso.rename({'lon_rho': 'lon', 'lat_rho': 'lat'})
    dso = dso.drop_vars(['pm', 'pn', 'Cs_r'])

    # Construct depth at rho points dataArray
    dso['zr'] = xr.DataArray(
        grd.vgrid.z_r[:], dims=['s_rho', 'eta_rho', 'xi_rho'],
        coords=dict(s_rho=grd.vgrid.s_r,
                    eta_rho=dso.eta_rho, xi_rho=dso.xi_rho),
        name='z_r')
    dso['dz'] = xr.DataArray(
        grd.vgrid.dz[:], dims=['s_rho', 'eta_rho', 'xi_rho'],
        coords=dict(s_rho=grd.vgrid.s_r,
                    eta_rho=dso.eta_rho, xi_rho=dso.xi_rho),
        name='dz')

    # If flood the raw grid, construct Mask for inputting dataset
    if flood_raw:
        regridder = xe.Regridder(dso, dsi, 'conservative')
        dsi.coords['mask_rho'] = regridder(dso.mask_rho.astype(np.float))
        dsi.coords['mask_rho'] = dsi.mask_rho.where(dsi.mask_rho < 0.01, 1)
        dsi.coords['mask_rho'] = dsi.mask_rho.where(dsi.mask_rho == 1, 0)

        dsi = dsi.drop_vars(['lon_b', 'lat_b'])
        dso = dso.drop_vars(['xi_vert', 'eta_vert', 'x_vert', 'y_vert',
                             'lon_b', 'lat_b'])

    return dsi, dso


def interp(dsi: xr.DataArray, dso: xr.DataArray,
           flood: bool = False, flood_raw: bool = False,
           sigma: float = 3.):
    """
    Perform interpolation on horizontal and vertical direction from input
    grid dsi to output grid dso.

    Usage:
        dsr = data_prep(dsi, dso,
                        flood=True, sigma=3)

    Inputs:
        dsi        - Xarray dataset containing the raw data
        dso        - Xarray dataset of the regridding grid
        flood      - boolean, if flood the regridded data
        flood_raw  - boolean, if flood the raw data
        sigma      - float, when flood the grid, sigma is used as a parameter
                     in the Gaussian filter (scipy.ndimage.gaussian_filter).

    outputs:
        dsr        - Xarray dataset of the reformatted raw data
    """

    # If dimension 'z' exists, need to extrapolate to deep water
    if 'z' in dsi.dims:
        dsi = dsi.interpolate_na(dim='z', method='nearest',
                                 fill_value='extrapolate')

    # Perform grid flooding.
    if flood and flood_raw:
        for var in dsi:
            var_data = dsi[var].values
            var_data = _flood_gaussian(var_data, sigma=sigma,
                                       mask_rho=dsi.mask_rho.data)
            dsi[var].values = var_data

    # Perform two-step horizontal/vertical interpolation w/ flooded data.
    # Horizontal interpolation
    regridder = xe.Regridder(dsi, dso, 'bilinear', unmapped_to_nan=True)
    dsr = regridder(dsi)
    dsr.coords['eta_rho'], dsr.coords['xi_rho'] = \
        dso.coords['eta_rho'], dso.coords['xi_rho']
    dsr = dsr.drop_vars(['lon', 'lat'])

    # If there are still unmaped points, perform a second round of grid
    # flooding. This often happends when the target grid is outside the
    # raw data domain.
    if flood:
        for var in dsr:
            if not dsr[var].where(dso.mask_rho == 1, 0).isnull().any():
                break
            var_data = dsr[var].values
            var_data = _flood_gaussian(var_data, sigma=sigma,
                                       mask_rho=dso.mask_rho.data)
            dsr[var].values = var_data

    # Vertical interpolation
    dsr.coords['xi_rho'], dsr.coords['eta_rho'] = dso.zr.xi_rho, dso.zr.eta_rho
    if 'z' in dsi.dims:
        dsr = dsr.interp({'z': -dso.zr}, kwargs={'fill_value': None})

    return dsr


def _flood_gaussian(var_data, sigma=3., mask_rho=None, quick=False):
    """
    Apply an Gaussian filter to flood the grid.

    Usage:
        out_data = _flood_gaussian(var_data, sigma=3.,
                                   mask_rho=None, quick=False)

    Inputs:
        var_data - numpy Array, data to be flooded. The array can be of
                   any dimension, but the last two dimensions has to be
                   [y(lat) and x(lon)] or [x(lon) and y(lat)].
        sigma    - float, when flood the grid, sigma is used as a parameter
                   in the Gaussian filter (scipy.ndimage.gaussian_filter).
        mask_rho - numpy Array, masks of land/ocean of in_data. If provided,
                   this array has to be 2-D of y(lat) and x(lon).
        quick    - boolean, if True, the function will not use Gaussian
                   convolution to compute flooded value. Instead it will use
                   average value of the 4 neighbor points to compute the
                   flooded value. This is not recommended, though.

    outputs:
        out_data - numpy Array, flooded data. out_data has the same dimension
                   as var_data.
    """

    # Get dimensions of the input data and construct indexers.
    ndim = var_data.ndim
    yn, xn = var_data.shape[-2:]
    slc = ()
    slc0 = ()
    pad_width = ()
    for i in range(ndim-2):
        slc = slc + (slice(None),)
        slc0 = slc0 + (0,)
        pad_width = pad_width + ((0, 0),)
    pad_width = pad_width + ((1, 1), (1, 1))

    # Initial mask values for comparison
    maski = np.ones(var_data.shape)
    while True:
        # Masks of valid data
        mask0 = ~np.isnan(var_data)

        # Find points to be flooded in this loop
        mask0nei = np.pad(mask0, pad_width, mode='edge')
        mask0nei = \
            mask0nei[slc + (slice(-2), slice(-2))] | \
            mask0nei[slc + (slice(2, None), slice(-2))] | \
            mask0nei[slc + (slice(-2), slice(2, None))] | \
            mask0nei[slc + (slice(2, None), slice(2, None))]
        mask = ~mask0 & mask0nei

        # Set invalid data to 0
        var_data0 = var_data.copy()
        var_data0[~mask0] = 0

        # Get values of flooded points
        if quick:
            var_data0 = np.pad(var_data0, pad_width, mode='edge')
            var_data0 = np.ma.masked_array([
                var_data0[slc + (slice(-2), slice(-2))],
                var_data0[slc + (slice(2, None), slice(-2))],
                var_data0[slc + (slice(-2), slice(2, None))],
                var_data0[slc + (slice(2, None), slice(2, None))]])
            var_data0 = np.nanmean(var_data0, axis=0)
        else:
            var_data0 = gaussian_filter1d(var_data0, sigma=sigma, axis=-1)
            var_data0 = gaussian_filter1d(var_data0, sigma=sigma, axis=-2)
            zero_filt = gaussian_filter(mask0[slc0]*1., sigma=sigma)
            var_data0 = var_data0 / zero_filt

        # Update values of flooded points and reset the mask
        var_data[mask] = var_data0[mask]

        if mask_rho is not None:
            mask = mask & (mask_rho == 1)
        if (mask == maski).all():
            break
        maski = mask

    if mask_rho is not None:
        # Flood one more time to fill the edges
        mask0 = ~np.isnan(var_data)
        mask0nei = np.pad(mask0, pad_width, mode='edge')
        mask0nei = \
            mask0nei[slc + (slice(-2), slice(-2))] | \
            mask0nei[slc + (slice(2, None), slice(-2))] | \
            mask0nei[slc + (slice(-2), slice(2, None))] | \
            mask0nei[slc + (slice(2, None), slice(2, None))]
        mask = ~mask0 & mask0nei
        var_data0 = var_data.copy()
        var_data0[~mask0] = 0
        if quick:
            var_data0 = np.ma.masked_array([
                var_data0[slc + (slice(-2), slice(-2))],
                var_data0[slc + (slice(2, None), slice(-2))],
                var_data0[slc + (slice(-2), slice(2, None))],
                var_data0[slc + (slice(2, None), slice(2, None))]])
            var_data0 = np.nanmean(axis=0)
        else:
            var_data0 = gaussian_filter1d(var_data0, sigma=sigma, axis=-1)
            var_data0 = gaussian_filter1d(var_data0, sigma=sigma, axis=-2)
            zero_filt = gaussian_filter(mask0[slc0]*1., sigma=sigma)
            var_data0 = var_data0 / zero_filt
        var_data[mask] = var_data0[mask]

    return var_data


def _flood_dask(var_data, dim):
    return xr.apply_ufunc(_flood_gaussian, var_data,
                          sigma=3., mask_rho=None, quick=False,
                          input_cor_dims=['lon', 'lat'],
                          dask='parallelized',
                          output_dtypes=float)


"""

Some obsolete flooding functions that didn't work well.

def __flood_wrapper(dsi, seed, cwid):

    # Fetch dimension lengths.
    dims = dsi.dims
    yn, xn = dims['lat'], dims['lon']

    var_list = [var for var in dsi]
    # Define 'seed' index to begin the flood operation.
    if seed is None:
        bottom_seed = \
            dsi[var_list[0]][{'ocean_time': 0, 'z': -1}].notnull()
        yseed, xseed = np.where(bottom_seed)
        cseed = np.hypot(yseed - yseed.mean(),
                         xseed - xseed.mean()).argmin()
        yseed, xseed = yseed[cseed], xseed[cseed]
    elif isinstance(seed, str):
        if seed.lower() == 'c':
            yseed, xseed = int((yn-1)/2), int((xn-1)/2)
        elif seed.lower() == 'w':
            yseed, xseed = int((yn-1)/2), 0
        elif seed.lower() == 'e':
            yseed, xseed = int((yn-1)/2), xn-1
        elif seed.lower() == 's':
            yseed, xseed = 0, int((xn-1)/2)
        elif seed.lower() == 'n':
            yseed, xseed = yn-1, int((xn-1)/2)
        elif seed.lower() in ['sw', 'ws']:
            yseed, xseed = 0, 0
        elif seed.lower() in ['nw', 'wn']:
            yseed, xseed = yn-1, 0
        elif seed.lower() in ['se', 'es']:
            yseed, xseed = 0, xn-1
        elif seed.lower() in ['ne', 'en']:
            yseed, xseed = yn-1, xn-1
    else:
        yseed, xseed = \
            max(0, min(yn-1, seed[0])), max(0, min(xn-1, seed[1]))

    # Loop over all variables to perform flooding.
    for var in dsi:
        var_data = dsi[var].values.copy()
        if 'ocean_time' not in dsi[var].dims:
            var_data = var_data[np.newaxis]
        if 'z' not in dsi[var].dims:
            var_data = var_data[:, np.newaxis]
        var_data = __flood_seed(var_data, yseed=yseed, xseed=xseed,
                                cwid=cwid)
        if 'z' not in dsi[var].dims:
            var_data = var_data[:, 0]
        if 'ocean_time' not in dsi[var].dims:
            var_data = var_data[0]
        dsi[var].values = var_data

    return dsi


def __flood_seed(var_data, yseed, xseed, cwid):

    # Get dimensions
    tn, zn, yn, xn = var_data.shape

    # Construct a numpy array to identify NaNs.
    is_nan = np.isnan(var_data)

    # Expand the data edges by cwid and mask NaNs.
    var_data = np.pad(var_data,
                      ((0, 0), (0, 0), (cwid, cwid), (cwid, cwid)),
                      mode='edge')
    var_data = np.ma.masked_invalid(var_data)

    yseedc, xseedc = yseed + cwid, xseed + cwid

    if is_nan[0, 0, yseed, xseed]:
        # If seed is NaN, calculate it based on revert distance.
        print('Seed value is NaN. Use distance-weighted mean.')
        print('Seed index: %d, %d' % (yseed, xseed))
        xx, yy = np.meshgrid(np.arange(xn), np.arange(yn))
        dist = np.hypot(xx - yseed, yy - xseed)
        dist = np.ma.masked_where(is_nan[0, 0], dist)
        dist_weight = np.exp(-dist/10)
        dist_weight = dist_weight/dist_weight.sum()
        var_data[:, :, yseedc, xseedc] = \
            (var_data[:, :]*dist_weight).sum(axis=(-2, -1))

    # Flood the grid from seed. For each NaN value the neighboring
    # points are used to calculat the average.
    for ci in range(1, max(xn-xseed, xseed, yn-yseed, yseed)):
        x0, x1, y0, y1 = xseedc-ci, xseedc+ci, yseedc-ci, yseedc+ci
        if y0 >= cwid:
            x00, x11 = max(x0, cwid), min(x1, xn+cwid)
            edg_mask = var_data.mask[:, :, y0, x00:x11]
            if cwid == 1:
                edg_data = np.ma.masked_array([
                    var_data[:, :, y0-1, x00:x11],
                    var_data[:, :, y0+1, x00:x11],
                    var_data[:, :, y0, x00-1:x11-1],
                    var_data[:, :, y0, x00+1:x11+1],
                    var_data[:, :, y0-1, x00-1:x11-1],
                    var_data[:, :, y0-1, x00+1:x11+1],
                    var_data[:, :, y0+1, x00-1:x11-1],
                    var_data[:, :, y0+1, x00+1:x11+1]]).mean(axis=0)
            else:
                edg_data = np.zeros((tn, zn, x11-x00))
                for i, xi in enumerate(range(x00, x11)):
                    if edg_mask[0, 0, i]:
                        edg_data[:, :, i] = \
                            var_data[
                                :, :,
                                y0-cwid:y0+cwid+1,
                                xi-cwid:xi+cwid+1].mean(axis=(-2, -1))
            var_data[:, :, y0, x00:x11][edg_mask] = edg_data[edg_mask]
        if y1 <= yn+cwid-1:
            x00, x11 = max(x0, cwid-1)+1, min(x1, xn+cwid-1)+1
            edg_mask = var_data.mask[:, :, y1, x00:x11]
            if cwid == 1:
                edg_data = np.ma.masked_array([
                    var_data[:, :, y1-1, x00:x11],
                    var_data[:, :, y1+1, x00:x11],
                    var_data[:, :, y1, x00-1:x11-1],
                    var_data[:, :, y1, x00+1:x11+1],
                    var_data[:, :, y1-1, x00-1:x11-1],
                    var_data[:, :, y1-1, x00+1:x11+1],
                    var_data[:, :, y1+1, x00-1:x11-1],
                    var_data[:, :, y1+1, x00+1:x11+1]]).mean(axis=0)
            else:
                edg_data = np.zeros((tn, zn, x11-x00))
                for i, xi in enumerate(range(x00, x11)):
                    if edg_mask[0, 0, i]:
                        edg_data[:, :, i] = \
                            var_data[
                                :, :,
                                y1-cwid:y1+cwid+1,
                                xi-cwid:xi+cwid+1].mean(axis=(-2, -1))
            var_data[:, :, y1, x00:x11][edg_mask] = edg_data[edg_mask]
        if x0 >= cwid:
            y00, y11 = max(y0, cwid-1)+1, min(y1, yn+cwid-1)+1
            edg_mask = var_data.mask[:, :, y00:y11, x0]
            if cwid == 1:
                edg_data = np.ma.masked_array([
                    var_data[:, :, y00:y11, x0-1],
                    var_data[:, :, y00:y11, x0+1],
                    var_data[:, :, y00-1:y11-1, x0],
                    var_data[:, :, y00+1:y11+1, x0],
                    var_data[:, :, y00-1:y11-1, x0-1],
                    var_data[:, :, y00+1:y11+1, x0-1],
                    var_data[:, :, y00-1:y11-1, x0+1],
                    var_data[:, :, y00+1:y11+1, x0+1]]).mean(axis=0)
            else:
                edg_data = np.zeros((tn, zn, y11-y00))
                for i, yi in enumerate(range(y00, y11)):
                    if edg_mask[0, 0, i]:
                        edg_data[:, :, i] = \
                            var_data[
                                :, :,
                                yi-cwid:yi+cwid+1,
                                x0-cwid:x0+cwid+1].mean(axis=(-2, -1))
            var_data[:, :, y00:y11, x0][edg_mask] = edg_data[edg_mask]
        if x1 <= xn+cwid-1:
            y00, y11 = max(y0, cwid), min(y1, yn+cwid)
            edg_mask = var_data.mask[:, :, y00:y11, x1]
            if cwid == 1:
                edg_data = np.ma.masked_array([
                    var_data[:, :, y00:y11, x1-1],
                    var_data[:, :, y00:y11, x1+1],
                    var_data[:, :, y00-1:y11-1, x1],
                    var_data[:, :, y00+1:y11+1, x1],
                    var_data[:, :, y00-1:y11-1, x1-1],
                    var_data[:, :, y00+1:y11+1, x1-1],
                    var_data[:, :, y00-1:y11-1, x1+1],
                    var_data[:, :, y00+1:y11+1, x1+1]]).mean(axis=0)
            else:
                edg_data = np.zeros((tn, zn, y11-y00))
                for i, yi in enumerate(range(y00, y11)):
                    if edg_mask[0, 0, i]:
                        edg_data[:, :, i] = \
                            var_data[
                                :, :,
                                yi-cwid:yi+cwid+1,
                                x1-cwid:x1+cwid+1].mean(axis=(-2, -1))
            var_data[:, :, y00:y11, x1][edg_mask] = edg_data[edg_mask]

    var_data = var_data[:, :, cwid:-cwid, cwid:-cwid]
    return var_data
"""
