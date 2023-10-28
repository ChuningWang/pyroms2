"""A thin wrapper for netCDF4.Dataset and netCDF4.MFDataset

This module provides two functions, Dataset and MFDataset, that are similar
to the netCDF4 functions of the same name. This package is a thin wrapper
around these functions, and provides two services. It will pass through
netCDF4 objects unchanged, so that netCDF objects, filenames, lists of files,
or strings with wildcards can be passed to the function indescriminately.

Examples of usage
-----------------

with an input of a string:
    # returns netCDF4.Dataset object based on file
    nc = pyroms.io.Dataset(file)

    # returns MFnetCDF4.Dataset object based on file (with wildcard chars)
    nc = pyroms.io.MFDataset(file)

with an input of a list of files:
    # returns MFDataset object based on list of files
    nc = pyroms.io.Dataset(files)

    # returns MFDataset object based on list of files
    nc = pyroms.io.MFDataset(files)

with an input of a netCDF4.Dataset or MFnetCDF4.Dataset object:
    # passes through netCDF4.Dataset or MFnetCDF4.Dataset object
    nc = pyroms.io.Dataset(nc)

    # passes through MFDataset object based on file (with wildcard chars)
    nc = pyroms.io.MFDataset(nc)
"""

from glob import glob
import numpy as np
from datetime import datetime
from numpy import datetime64
import netCDF4 as nc


def Dataset(ncfile):
    """
    Return an appropriate netcdf object:
        netCDF4 object given a file string
        MFnetCDF4 object given a list of files

    A netCDF4 or MFnetCDF4 object returns itself
    """
    if isinstance(ncfile, str):
        return nc.Dataset(ncfile, 'r')
    elif isinstance(ncfile, list) or isinstance(ncfile, tuple):
        return nc.MFDataset(sorted(ncfile))
    # accept any oject with a variables attribute
    elif hasattr(ncfile, 'variables'):
        assert isinstance(ncfile.variables, dict), \
               'variables attribute must be a dictionary'
        return ncfile
    else:
        raise TypeError('type %s not supported' % type(ncfile))


Dataset.__doc__ = __doc__


def MFDataset(ncfile):
    """
    Return an MFnetCDF4 object given a string or list. A string is
       expanded with wildcards using glob. A netCDF4 or MFnetCDF4 object
       returns itself.
    """
    if isinstance(ncfile, str):
        ncfiles = glob(ncfile)
        return nc.MFDataset(sorted(ncfiles))
    elif isinstance(ncfile, list) or isinstance(ncfile, tuple):
        return nc.MFDataset(sorted(ncfile))
    # accept any oject with a variables attribute
    elif hasattr(ncfile, 'variables'):
        assert isinstance(ncfile.variables, dict), \
               'variables attribute must be a dictionary'
        return ncfile
    else:
        raise TypeError('type %s not supported' % type(ncfile))
        return nc.MFDataset(ncfiles)


MFDataset.__doc__ = __doc__


def nc_create_var(fh, name, dimensions,
                  long_name=None, units=None, field=None, fill_value=None):
    fh.createVariable(name, 'f8', dimensions, fill_value=fill_value)
    if long_name is not None:
        fh.variables[name].long_name = long_name
    if units is not None:
        fh.variables[name].units = units
    if field is not None:
        fh.variables[name].field = field


def nc_write_var(fh, var, name, dimensions,
                 long_name=None, units=None, field=None, fill_value=None):
    nc_create_var(fh, name, dimensions, long_name, units, field, fill_value)
    fh.variables[name][:] = var
    print(' ... wrote ', name)


def nc_create_roms_file(filename, grd, tref=None,
                        lgrid=True, geogrid=True):
    """
    Create ROMS initial file.
    """

    fh = nc.Dataset(filename, 'w')
    fh.Description = 'ROMS file'
    fh.Author = 'pyroms.io.nc_create_roms_file'
    fh.Created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fh.title = 'ROMS file'

    fh.createDimension('xi_rho', np.size(grd.hgrid.mask_rho, 1))
    fh.createDimension('xi_u', np.size(grd.hgrid.mask_u, 1))
    fh.createDimension('xi_v', np.size(grd.hgrid.mask_v, 1))
    fh.createDimension('xi_psi', np.size(grd.hgrid.mask_psi, 1))
    fh.createDimension('eta_rho', np.size(grd.hgrid.mask_rho, 0))
    fh.createDimension('eta_u', np.size(grd.hgrid.mask_u, 0))
    fh.createDimension('eta_v', np.size(grd.hgrid.mask_v, 0))
    fh.createDimension('eta_psi', np.size(grd.hgrid.mask_psi, 0))
    fh.createDimension('s_rho', grd.vgrid.N)
    fh.createDimension('s_w', grd.vgrid.Np)
    fh.createDimension('ocean_time', None)

    # write time and grid information
    fh.createVariable('theta_s', 'f8', ())
    fh.variables['theta_s'].long_name = \
        'S-coordinate surface control parameter'
    fh.variables['theta_s'][:] = grd.vgrid.theta_s

    fh.createVariable('theta_b', 'f8', ())
    fh.variables['theta_b'].long_name = 'S-coordinate bottom control parameter'
    fh.variables['theta_b'][:] = grd.vgrid.theta_b

    fh.createVariable('Tcline', 'f8', ())
    fh.variables['Tcline'].long_name = 'S-cordinate surface/bottom layer width'
    fh.variables['Tcline'].units = 'meter'
    fh.variables['Tcline'][:] = grd.vgrid.Tcline

    fh.createVariable('hc', 'f8', ())
    fh.variables['hc'].long_name = 'S-coordinate parameter, critical depth'
    fh.variables['hc'].units = 'meter'
    fh.variables['hc'][:] = grd.vgrid.hc

    fh.createVariable('s_rho', 'f8', ('s_rho'))
    fh.variables['s_rho'].long_name = 'S-coordinate at RHO-points'
    fh.variables['s_rho'].valid_min = -1.0
    fh.variables['s_rho'].valid_max = 0.0
    fh.variables['s_rho'].field = 's_rho,scalar'
    fh.variables['s_rho'][:] = grd.vgrid.s_r

    fh.createVariable('s_w', 'f8', ('s_w'))
    fh.variables['s_w'].long_name = 'S-coordinate at W-points'
    fh.variables['s_w'].valid_min = -1.0
    fh.variables['s_w'].valid_max = 0.0
    fh.variables['s_w'].field = 's_w,scalar'
    fh.variables['s_w'][:] = grd.vgrid.s_w

    fh.createVariable('Cs_r', 'f8', ('s_rho'))
    fh.variables['Cs_r'].long_name = \
        'S-coordinate stretching curves at RHO-points'
    fh.variables['Cs_r'].valid_min = -1.0
    fh.variables['Cs_r'].valid_max = 0.0
    fh.variables['Cs_r'].field = 'Cs_r,scalar'
    fh.variables['Cs_r'][:] = grd.vgrid.Cs_r

    fh.createVariable('Cs_w', 'f8', ('s_w'))
    fh.variables['Cs_w'].long_name = \
        'S-coordinate stretching curves at W-points'
    fh.variables['Cs_w'].valid_min = -1.0
    fh.variables['Cs_w'].valid_max = 0.0
    fh.variables['Cs_w'].field = 'Cs_w,scalar'
    fh.variables['Cs_w'][:] = grd.vgrid.Cs_w

    if (lgrid):
        fh.createVariable('h', 'f8', ('eta_rho', 'xi_rho'))
        fh.variables['h'].long_name = 'bathymetry at RHO-points'
        fh.variables['h'].units = 'meter'
        fh.variables['h'].coordinates = 'lon_rho lat_rho'
        fh.variables['h'].field = 'bath, scalar'
        fh.variables['h'][:] = grd.vgrid.h

        fh.createVariable('pm', 'f8', ('eta_rho', 'xi_rho'))
        fh.variables['pm'].long_name = 'curvilinear coordinate metric in XI'
        fh.variables['pm'].units = 'meter-1'
        fh.variables['pm'].coordinates = 'lon_rho lat_rho'
        fh.variables['pm'].field = 'pm, scalar'
        fh.variables['pm'][:] = grd.hgrid.pm

        fh.createVariable('pn', 'f8', ('eta_rho', 'xi_rho'))
        fh.variables['pn'].long_name = 'curvilinear coordinate metric in ETA'
        fh.variables['pn'].units = 'meter-1'
        fh.variables['pn'].coordinates = 'lon_rho lat_rho'
        fh.variables['pn'].field = 'pn, scalar'
        fh.variables['pn'][:] = grd.hgrid.pn

        if (geogrid):
            fh.createVariable('lon_rho', 'f8', ('eta_rho', 'xi_rho'))
            fh.variables['lon_rho'].long_name = 'longitude of RHO-points'
            fh.variables['lon_rho'].units = 'degree_east'
            fh.variables['lon_rho'].field = 'lon_rho, scalar'
            fh.variables['lon_rho'][:] = grd.hgrid.lon_rho

            fh.createVariable('lat_rho', 'f8', ('eta_rho', 'xi_rho'))
            fh.variables['lat_rho'].long_name = 'latitude of RHO-points'
            fh.variables['lat_rho'].units = 'degree_north'
            fh.variables['lat_rho'].field = 'lat_rho, scalar'
            fh.variables['lat_rho'][:] = grd.hgrid.lat_rho

            fh.createVariable('lon_u', 'f8', ('eta_u', 'xi_u'))
            fh.variables['lon_u'].long_name = 'longitude of U-points'
            fh.variables['lon_u'].units = 'degree_east'
            fh.variables['lon_u'].field = 'lon_u, scalar'
            fh.variables['lon_u'][:] = grd.hgrid.lon_u

            fh.createVariable('lat_u', 'f8', ('eta_u', 'xi_u'))
            fh.variables['lat_u'].long_name = 'latitude of U-points'
            fh.variables['lat_u'].units = 'degree_north'
            fh.variables['lat_u'].field = 'lat_u, scalar'
            fh.variables['lat_u'][:] = grd.hgrid.lat_u

            fh.createVariable('lon_v', 'f8', ('eta_v', 'xi_v'))
            fh.variables['lon_v'].long_name = 'longitude of V-points'
            fh.variables['lon_v'].units = 'degree_east'
            fh.variables['lon_v'].field = 'lon_v, scalar'
            fh.variables['lon_v'][:] = grd.hgrid.lon_v

            fh.createVariable('lat_v', 'f8', ('eta_v', 'xi_v'))
            fh.variables['lat_v'].long_name = 'latitude of V-points'
            fh.variables['lat_v'].units = 'degree_north'
            fh.variables['lat_v'].field = 'lat_v, scalar'
            fh.variables['lat_v'][:] = grd.hgrid.lat_v

            fh.createVariable('lon_psi', 'f8', ('eta_psi', 'xi_psi'))
            fh.variables['lon_psi'].long_name = 'longitude of PSI-points'
            fh.variables['lon_psi'].units = 'degree_east'
            fh.variables['lon_psi'].field = 'lon_psi, scalar'
            fh.variables['lon_psi'][:] = grd.hgrid.lon_psi

            fh.createVariable('lat_psi', 'f8', ('eta_psi', 'xi_psi'))
            fh.variables['lat_psi'].long_name = 'latitude of PSI-points'
            fh.variables['lat_psi'].units = 'degree_north'
            fh.variables['lat_psi'].field = 'lat_psi, scalar'
            fh.variables['lat_psi'][:] = grd.hgrid.lat_psi
        else:
            fh.createVariable('x_rho', 'f8', ('eta_rho', 'xi_rho'))
            fh.variables['x_rho'].long_name = 'x location of RHO-points'
            fh.variables['x_rho'].units = 'meter'
            fh.variables['x_rho'].field = 'x_rho, scalar'
            fh.variables['x_rho'][:] = grd.hgrid.x_rho

            fh.createVariable('y_rho', 'f8', ('eta_rho', 'xi_rho'))
            fh.variables['y_rho'].long_name = 'y location of RHO-points'
            fh.variables['y_rho'].units = 'meter'
            fh.variables['y_rho'].field = 'y_rho, scalar'
            fh.variables['y_rho'][:] = grd.hgrid.y_rho

            fh.createVariable('x_u', 'f8', ('eta_u', 'xi_u'))
            fh.variables['x_u'].long_name = 'x location of U-points'
            fh.variables['x_u'].units = 'meter'
            fh.variables['x_u'].field = 'x_u, scalar'
            fh.variables['x_u'][:] = grd.hgrid.x_u

            fh.createVariable('y_u', 'f8', ('eta_u', 'xi_u'))
            fh.variables['y_u'].long_name = 'y location of U-points'
            fh.variables['y_u'].units = 'meter'
            fh.variables['y_u'].field = 'y_u, scalar'
            fh.variables['y_u'][:] = grd.hgrid.y_u

            fh.createVariable('x_v', 'f8', ('eta_v', 'xi_v'))
            fh.variables['x_v'].long_name = 'x location of V-points'
            fh.variables['x_v'].units = 'meter'
            fh.variables['x_v'].field = 'x_v, scalar'
            fh.variables['x_v'][:] = grd.hgrid.x_v

            fh.createVariable('y_v', 'f8', ('eta_v', 'xi_v'))
            fh.variables['y_v'].long_name = 'y location of V-points'
            fh.variables['y_v'].units = 'meter'
            fh.variables['y_v'].field = 'y_v, scalar'
            fh.variables['y_v'][:] = grd.hgrid.y_v

            fh.createVariable('x_psi', 'f8', ('eta_psi', 'xi_psi'))
            fh.variables['x_psi'].long_name = 'x location of PSI-points'
            fh.variables['x_psi'].units = 'meter'
            fh.variables['x_psi'].field = 'x_psi, scalar'
            fh.variables['x_psi'][:] = grd.hgrid.x_psi

            fh.createVariable('y_psi', 'f8', ('eta_psi', 'xi_psi'))
            fh.variables['y_psi'].long_name = 'y location of PSI-points'
            fh.variables['y_psi'].units = 'meter'
            fh.variables['y_psi'].field = 'y_psi, scalar'
            fh.variables['y_psi'][:] = grd.hgrid.y_psi

        fh.createVariable('angle', 'f8', ('eta_rho', 'xi_rho'))
        fh.variables['angle'].long_name = 'angle between XI-axis and EAST'
        fh.variables['angle'].units = 'radians'
        fh.variables['angle'].coordinates = 'lon_rho lat_rho'
        fh.variables['angle'].field = 'angle, scalar'
        fh.variables['angle'][:] = grd.hgrid.angle_rho

        fh.createVariable('mask_rho', 'f8', ('eta_rho', 'xi_rho'))
        fh.variables['mask_rho'].long_name = 'mask on RHO-points'
        fh.variables['mask_rho'].option_0 = 'land'
        fh.variables['mask_rho'].option_1 = 'water'
        fh.variables['mask_rho'].coordinates = 'lon_rho lat_rho'
        fh.variables['mask_rho'][:] = grd.hgrid.mask_rho

        fh.createVariable('mask_u', 'f8', ('eta_u', 'xi_u'))
        fh.variables['mask_u'].long_name = 'mask on U-points'
        fh.variables['mask_u'].option_0 = 'land'
        fh.variables['mask_u'].option_1 = 'water'
        fh.variables['mask_u'].coordinates = 'lon_u lat_u'
        fh.variables['mask_u'][:] = grd.hgrid.mask_u

        fh.createVariable('mask_v', 'f8', ('eta_v', 'xi_v'))
        fh.variables['mask_v'].long_name = 'mask on V-points'
        fh.variables['mask_v'].option_0 = 'land'
        fh.variables['mask_v'].option_1 = 'water'
        fh.variables['mask_v'].coordinates = 'lon_v lat_v'
        fh.variables['mask_v'][:] = grd.hgrid.mask_v

        fh.createVariable('mask_psi', 'f8', ('eta_psi', 'xi_psi'))
        fh.variables['mask_psi'].long_name = 'mask on PSI-points'
        fh.variables['mask_psi'].option_0 = 'land'
        fh.variables['mask_psi'].option_1 = 'water'
        fh.variables['mask_psi'].coordinates = 'lon_psi lat_psi'
        fh.variables['mask_psi'][:] = grd.hgrid.mask_psi

    tformat = '%Y-%m-%d %H:%M:%S'
    if tref is None:
        units = 'seconds since 1900-01-01 00:00:00'
    elif isinstance(tref, datetime):
        units = 'seconds since ' + tref.strftime(tformat)
    elif isinstance(tref, datetime64):
        units = 'seconds since ' + tref.item().strftime(tformat)
    elif isinstance(tref, str):
        units = 'seconds since ' + datetime64(tref).item().strftime(tformat)

    nc_create_var(fh, 'ocean_time', ('ocean_time'),
                  'time since initialization', units, 'time, scalar, series')
    fh.variables['ocean_time'].calendar = 'proleptic_gregorian'

    fh.close()


def nc_create_roms_bdry_file(filename, grd, tref=None):
    """
    Create ROMS boundary file.

    var_name_full = var_name + '_' + var_dir
    var_long_name_full = var_long_name + ' ' + var_dir + 'boundary condition'
    if var_dir in ['north', 'south']:
        var_dim = ('ocean_time', 'xi_rho')
    else:
        var_dim = ('ocean_time', 'eta_rho')
    var_field = var_name_full + ', scalar, series'
    """

    # create file
    fh = nc.Dataset(filename, 'w')
    fh.Description = 'ROMS boundary file'
    fh.Author = 'pyroms.io.nc_create_roms_bdry_file'
    fh.Created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fh.title = 'ROMS file'

    eta_rho, xi_rho = grd.hgrid.mask_rho.shape
    eta_u, xi_u = grd.hgrid.mask_u.shape
    eta_v, xi_v = grd.hgrid.mask_v.shape

    fh.createDimension('xi_rho', xi_rho)
    fh.createDimension('eta_rho', eta_rho)
    fh.createDimension('xi_u', xi_u)
    fh.createDimension('eta_u', eta_u)
    fh.createDimension('xi_v', xi_v)
    fh.createDimension('eta_v', eta_v)
    fh.createDimension('s_rho', grd.vgrid.N)
    fh.createDimension('ocean_time', None)

    tformat = '%Y-%m-%d %H:%M:%S'
    if tref is None:
        units = "seconds since 1900-01-01 00:00:00"
    elif isinstance(tref, datetime):
        units = 'seconds since ' + tref.strftime(tformat)
    elif isinstance(tref, datetime64):
        units = 'seconds since ' + tref.item().strftime(tformat)
    elif isinstance(tref, str):
        units = 'seconds since ' + datetime64(tref).item().strftime(tformat)

    nc_create_var(fh, 'ocean_time', ('ocean_time'),
                  'time since initialization', units, 'time, scalar, series')
    fh.variables['ocean_time'].calendar = 'proleptic_gregorian'

    fh.close()


def nc_create_roms_river_file(filename, grd, nriver, tref=None):
    """
    Create ROMS river file.

    Inputs:
        filename     - name of output netCDF file
        grd          - pyroms grid object
        nriver       - number of river points
    """

    # create file
    fh = nc.Dataset(filename, 'w')
    fh.Description = 'ROMS river file'
    fh.Author = 'pyroms.io.nc_create_roms_river_file'
    fh.Created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fh.title = 'ROMS file'

    fh.createDimension('river', nriver)
    fh.createDimension('s_rho', grd.vgrid.N)
    fh.createDimension('river_time', None)

    tformat = '%Y-%m-%d %H:%M:%S'
    if tref is None:
        units = "seconds since 1900-01-01 00:00:00"
    elif isinstance(tref, datetime):
        units = 'seconds since ' + tref.strftime(tformat)
    elif isinstance(tref, datetime64):
        units = 'seconds since ' + tref.item().strftime(tformat)
    elif isinstance(tref, str):
        units = 'seconds since ' + datetime64(tref).item().strftime(tformat)

    nc_create_var(fh, 'river_time', ('river_time'),
                  'time since initialization', units, 'time, scalar, series')
    fh.variables['river_time'].calendar = 'proleptic_gregorian'

    fh.close()


def nc_create_roms_tide_file(filename, grd, consts,
                             fill_value=None, write_coords=False):
    """
    Create ROMS tidal forcing file.

    Inputs:
        filename     - name of output netCDF file
        grd          - pyroms grid object
        fill_value   - fill_value parameter for netCDF4
        write_coords - boolean, if write lat/lon/mask in netCDF4 file
    """

    nconsts = len(consts)

    # Constituents information (from pyTMD)
    # -- constituents array that are included in tidal program
    cindex = ['m2', 's2', 'k1', 'o1', 'n2', 'p1', 'k2', 'q1', '2n2', 'mu2',
              'nu2', 'l2', 't2', 'j1', 'm1', 'oo1', 'rho1', 'mf', 'mm', 'ssa',
              'm4', 'ms4', 'mn4', 'm6', 'm8', 'mk3', 's6', '2sm2', '2mk3']
    # -- species type (spherical harmonic dependence of quadrupole potential)
    # species = np.array([2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    #                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # -- loading love number
    # -- alpha = correction factor for first order load tides
    # alpha = np.array([0.693, 0.693, 0.736, 0.695, 0.693, 0.706, 0.693,
    #                   0.695, 0.693, 0.693, 0.693, 0.693, 0.693, 0.695,
    #                   0.695, 0.695, 0.695, 0.693, 0.693, 0.693, 0.693,
    #                   0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693,
    #                   0.693])
    # -- omega: angular frequency of constituent, in radians
    omega = np.array([1.405189e-04, 1.454441e-04, 7.292117e-05, 6.759774e-05,
                      1.378797e-04, 7.252295e-05, 1.458423e-04, 6.495854e-05,
                      1.352405e-04, 1.355937e-04, 1.382329e-04, 1.431581e-04,
                      1.452450e-04, 7.556036e-05, 7.028195e-05, 7.824458e-05,
                      6.531174e-05, 0.053234e-04, 0.026392e-04, 0.003982e-04,
                      2.810377e-04, 2.859630e-04, 2.783984e-04, 4.215566e-04,
                      5.620755e-04, 2.134402e-04, 4.363323e-04, 1.503693e-04,
                      2.081166e-04])

    per = []
    for con in consts:
        if con.lower() in cindex:
            idx = cindex.index(con.lower())
            per.append(2.*np.pi/omega[idx]/3600.)
        else:
            assert 'Unknown tidal constitute: ' + con
    per = np.array(per)

    # create file
    fh = nc.Dataset(filename, 'w')
    fh.Description = 'ROMS tide file'
    fh.Author = 'pyroms.io.nc_create_roms_tide_file'
    fh.Created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fh.title = 'ROMS tide file'
    # fh.grid_file = grd.name

    eta_rho, xi_rho = grd.hgrid.mask_rho.shape
    eta_u, xi_u = grd.hgrid.mask_u.shape
    eta_v, xi_v = grd.hgrid.mask_v.shape

    fh.createDimension('namelen', 4)
    fh.createDimension('tide_period', nconsts)
    fh.createDimension('eta_rho', eta_rho)
    fh.createDimension('xi_rho', xi_rho)

    name_nc = fh.createVariable('tide_name', 'c', ('tide_period', 'namelen'))
    for i, ci in enumerate(consts):
        name_nc[i, :len(ci)] = list(ci)

    per_nc = fh.createVariable('tide_period', 'f8', ('tide_period'))
    per_nc.field = 'tide_period, scalar'
    per_nc.long_name = 'tidal angular period'
    per_nc.units = 'hours'
    per_nc[:] = per

    if write_coords:
        lat_nc = fh.createVariable('lat_rho', 'd', ('eta_rho', 'xi_rho'))
        lat_nc.field = 'lat_rho, scalar'
        lat_nc.long_name = 'latitude of RHO-points'
        lat_nc.units = 'degree north'
        lat_nc[:] = grd.hgrid.lat_rho

        lon_nc = fh.createVariable('lon_rho', 'd', ('eta_rho', 'xi_rho'))
        lon_nc.field = 'lon_rho, scalar'
        lon_nc.long_name = 'longitude of RHO-points'
        lon_nc.units = 'degree east'
        lon_nc[:] = grd.hgrid.lon_rho

        msk_nc = fh.createVariable('mask_rho', 'd', ('eta_rho', 'xi_rho'))
        msk_nc.long_name = 'mask on RHO-points'
        msk_nc.option_0 = 'land'
        msk_nc.option_1 = 'water'
        msk_nc[:] = grd.hgrid.mask_rho

    Eamp_nc = fh.createVariable('tide_Eamp', 'f8',
                                ('tide_period', 'eta_rho', 'xi_rho'),
                                fill_value=fill_value)
    Eamp_nc.field = 'tide_Eamp, scalar'
    Eamp_nc.long_name = 'tidal elevation amplitude'
    Eamp_nc.units = 'meter'

    Ephase_nc = fh.createVariable('tide_Ephase', 'f8',
                                  ('tide_period', 'eta_rho', 'xi_rho'),
                                  fill_value=fill_value)
    Ephase_nc.field = 'tide_Ephase, scalar'
    Ephase_nc.long_name = 'tidal elevation phase angle'
    Ephase_nc.units = 'degrees, time of maximum elevation with respect ' + \
                      'chosen time orgin'

    Cmax_nc = fh.createVariable('tide_Cmax', 'f8',
                                ('tide_period', 'eta_rho', 'xi_rho'),
                                fill_value=fill_value)
    Cmax_nc.field = 'tide_Cmax, scalar'
    Cmax_nc.long_name = 'maximum tidal current, ellipse semi-major axis'
    Cmax_nc.units = 'meter second-1'

    Cmin_nc = fh.createVariable('tide_Cmin', 'f8',
                                ('tide_period', 'eta_rho', 'xi_rho'),
                                fill_value=fill_value)
    Cmin_nc.field = 'tide_Cmin, scalar'
    Cmin_nc.long_name = 'minimum tidal current, ellipse semi-minor axis'
    Cmin_nc.units = 'meter second-1'

    Cangle_nc = fh.createVariable('tide_Cangle', 'f8',
                                  ('tide_period', 'eta_rho', 'xi_rho'),
                                  fill_value=fill_value)
    Cangle_nc.field = 'tide_Cangle, scalar'
    Cangle_nc.long_name = 'tidal current inclination angle'
    Cangle_nc.units = 'degrees between semi-major axis and East'

    Cphase_nc = fh.createVariable('tide_Cphase', 'f8',
                                  ('tide_period', 'eta_rho', 'xi_rho'),
                                  fill_value=fill_value)
    Cphase_nc.field = 'tide_Cphase, scalar'
    Cphase_nc.long_name = 'tidal current phase angle'
    Cphase_nc.units = 'degrees, time of maximum velocity'

    fh.close()


if __name__ == '__main__':
    pass
