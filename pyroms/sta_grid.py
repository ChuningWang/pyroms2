"""
Tools for creating and working with ROMS Station Grids.
"""

from typing import Union
import numpy as np
from datetime import datetime
import netCDF4 as nc
import pyproj

from pyroms.sta_hgrid import StaHGrid, StaHGridGeo
from pyroms.vgrid import SCoord, ZCoord
from pyroms import io


class StaGrid(object):
    """
    Stations Grid object combining horizontal and vertical grid

    grd = StaGrid(sta_hgrid, vgrid)
    """

    def __init__(self,
                 name: str,
                 hgrid: Union[StaHGrid, StaHGridGeo],
                 vgrid: Union[SCoord, ZCoord]):
        self.name = name
        if isinstance(hgrid, StaHGrid):
            self.hgrid = hgrid
        else:
            raise TypeError('hgrid must be StaHGrid or StaHGridGeo.')
        if isinstance(vgrid, [SCoord, ZCoord]):
            self.vgrid = vgrid
        else:
            raise TypeError('vgrid must be SCoord or ZCoord.')


def get_sta_hgrid(sta_file,
                  proj: Union[type(None), str, pyproj.Proj] = None):
    """
    Load Stations horizontal grid object

    hgrid = get_sta_hgrid(sta_file)
    """

    fh = io.Dataset(sta_file)

    # Check for cartesian or geographical grid
    spherical = fh.variables['spherical'][0].item()
    if spherical == 0 or spherical == 'F':
        spherical = False
    else:
        spherical = True

    # Get horizontal grid
    print('Load station coordinates from file')
    x = fh.variables['x'][:]
    y = fh.variables['y'][:]
    if 'angle' in fh.variables.keys():
        angle = fh.variables['angle'][:]
    else:
        angle = np.zeros(x.shape)

    if spherical:
        if 'lon' in list(fh.variables.keys()) and \
           'lat' in list(fh.variables.keys()):
            lon = fh.variables['lon'][:]
            lat = fh.variables['lat'][:]
        else:
            if isinstance(proj, pyproj.Proj):
                pproj = proj
            elif hasattr(proj, 'proj4string'):
                pproj = pyproj.Proj(proj.proj4string)
            elif isinstance(proj, str):
                pproj = pyproj.Proj(proj)
            else:
                raise ValueError('Projection transformer must be ' +
                                 'provided if x/y are missing.')
            lon, lat = pproj(x, y, inverse=True)
        # Construct geographical grid
        hgrd = StaHGridGeo(lon, lat, x, y, angle, proj)
    else:
        # Construct cartisian grid
        hgrd = StaHGrid(x, y, angle)

    fh.close()

    return hgrd


def get_sta_grid(gridid: str, sta_file: str,
                 zeta: Union[type(None), np.ndarray] = None):
    """
    grd = get_sta_grid(gridid, sta_file, zeta=None)

    Load Stations grid object.

    gridid is a string with the name of the grid in it.  sta_file
    is the name of a stations file to read.

    grd.vgrid is a SCoord or a ZCoord object, depending on gridid.grdtype.

    grd.vgrid.z_r and grd.vgrid.z_w (grd.vgrid.z for a
    z_coordinate object) can be indexed in order to retreive the
    actual depths. The free surface time series zeta can be provided
    as an optional argument. Note that the values of zeta are not
    calculated until z is indexed, so a netCDF variable for zeta may
    be passed, even if the file is large, as only the values that
    are required will be retrieved from the file.
    """
    from pyroms.grid import ROMSGridInfo, get_ROMS_vgrid

    gridinfo = ROMSGridInfo(gridid, grid_file=sta_file, hist_file=sta_file)
    name = gridinfo.name

    hgrd = get_sta_hgrid(sta_file)
    vgrid = get_ROMS_vgrid(sta_file, zeta=zeta)

    # Get station grid
    return StaGrid(name, hgrd, vgrid)


def write_sta_grid(grd: StaGrid, filename: str = 'roms_sta_grd.nc'):
    """
    Write Stations_CGrid class on a NetCDF file.

    write_Stations_grid(grd, filename)
    """

    sn = grd.hgrid.x.shape[0]

    # Write Stations grid to file
    fh = nc.Dataset(filename, 'w')
    fh.Description = 'Station grid'
    fh.Author = 'pyroms.sta_grid.write_grd'
    fh.Created = datetime.now().isoformat()
    fh.type = 'Stations grid file'
    if isinstance(grd.hgrid, StaHGridGeo):
        if isinstance(grd.hgrid.proj, pyproj.Proj):
            fh.proj4_init = grd.hgrid.proj.crs.to_wkt()
        elif hasattr(grd.hgrid.proj, 'proj4string'):
            fh.proj4_init = grd.hgrid.proj.proj4string
        else:
            fh.proj4_init = ''

    fh.createDimension('station', sn)

    if hasattr(grd.vgrid, 's_rho'):
        if grd.vgrid.s_r is not None:
            N, = grd.vgrid.s_rho.shape
            fh.createDimension('s_rho', N)
            fh.createDimension('s_w', N+1)

            io.nc_write_var(fh, grd.vgrid.theta_s, 'theta_s', (),
                            'S-coordinate surface control parameter')
            io.nc_write_var(fh, grd.vgrid.theta_b, 'theta_b', (),
                            'S-coordinate bottom control parameter')
            io.nc_write_var(fh, grd.vgrid.Tcline, 'Tcline', (),
                            'S-coordinate surface/bottom layer width', 'meter')
            io.nc_write_var(fh, grd.vgrid.hc, 'hc', (),
                            'S-coordinate parameter, critical depth', 'meter')
            io.nc_write_var(fh, grd.vgrid.s_rho, 's_rho', ('s_rho'),
                            'S-coordinate at RHO-points')
            io.nc_write_var(fh, grd.vgrid.s_w, 's_w', ('s_w'),
                            'S-coordinate at W-points')
            io.nc_write_var(fh, grd.vgrid.Cs_r, 'Cs_r', ('s_rho'),
                            'S-coordinate stretching curves at RHO-points')
            io.nc_write_var(fh, grd.vgrid.Cs_w, 'Cs_w', ('s_w'),
                            'S-coordinate stretching curves at W-points')
            if hasattr(grd.vgrid, 'Vtrans'):
                if grd.vgrid.Vtrans is not None:
                    io.nc_write_var(fh, grd.vgrid.Vtrans, 'Vtransform', (),
                                    'S-coordinate transformation parameter')
                    io.nc_write_var(fh, grd.vgrid.Vstretch, 'Vstretching', (),
                                    'S-coordinate stretching parameter')
    if hasattr(grd.hgrid, 'h'):
        io.nc_write_var(fh, grd.vgrid.h, 'h', ('station'),
                        'bathymetry at RHO-points', 'meter')
    if hasattr(grd.vgrid, 'zice') is True:
        io.nc_write_var(fh, grd.vgrid.zice, 'zice', ('station'),
                        'iceshelf depth at RHO-points', 'meter')

    io.nc_write_var(fh, grd.hgrid.x, 'x', ('station'),
                    'x location of RHO-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.y, 'y', ('station'),
                    'y location of RHO-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.angle, 'angle', ('station'),
                    'angle between XI-axis and EAST', 'radians')
    if hasattr(grd.hgrid, 'lon'):
        io.nc_write_var(fh, grd.hgrid.lon, 'lon', ('station'),
                        'longitude of RHO-points', 'degree_east')
        io.nc_write_var(fh, grd.hgrid.lat, 'lat', ('station'),
                        'latitude of RHO-points', 'degree_north')

    fh.createVariable('spherical', 'i')
    fh.variables['spherical'].long_name = 'Grid type logical switch'
    if hasattr(grd.hgrid, 'lon'):
        fh.variables['spherical'][:] = 1
    else:
        fh.variables['spherical'][:] = 0
    print(' ... wrote ', 'spherical')

    fh.close()
