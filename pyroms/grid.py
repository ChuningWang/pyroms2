import os
import yaml
import numpy as np
import xarray as xr
from datetime import datetime
import netCDF4 as nc
import pyproj

from pyroms.hgrid import CGrid, CGridGeo, rho_to_vert, rho_to_vert_geo
from pyroms.vgrid import SCoord, ZCoord
from pyroms.sta_grid import StaGrid
from pyroms import io

"""
define a dictionary that will remember gridid's that are defined from
a history and grid file. Because this is defined in this model's name
space, it will remain persistent.  The keys are the gridid, and the
values are ROMSGridInfo objects.
"""

gridid_dictionary = {}


class ROMSGrid(object):
    """
    ROMS Grid object combining horizontal and vertical grid

    grd = ROMSGrid(hgrid, vgrid)
    """

    def __init__(self, name, hgrid=CGrid, vgrid=SCoord):
        self.name = name
        self.hgrid = hgrid
        self.vgrid = vgrid

    def write_grid(self, filename='roms_grd.nc'):
        """
        Write ROMS grid to a file.
        """

        write_ROMS_grid(self, filename=filename)

    def to_xarray(self):
        """
        Convert ROMSGrid object to Xarray dataset.

        ds_grd = grd.xarray_ROMS_grid()
        """
        en, xn = self.hgrid.x_rho.shape
        ds_grd = xr.Dataset(
            data_vars=dict(
                h=(
                    ['eta_rho', 'xi_rho'], self.vgrid.h,
                    dict(long_name='bathymetry at RHO-points',
                         units='meter')),
                f=(
                    ['eta_rho', 'xi_rho'], self.hgrid.f,
                    dict(long_name='iceshelf depth at RHO-points',
                         units='meter')),
                angle=(
                    ['eta_rho', 'xi_rho'], self.hgrid.angle_rho,
                    dict(long_name='angle between XI-axis and EAST',
                         units='radians')),
                mask_rho=(
                    ['eta_rho', 'xi_rho'], self.hgrid.mask_rho,
                    dict(long_name='mask on RHO-points')),
                mask_psi=(
                    ['eta_psi', 'xi_psi'], self.hgrid.mask_psi,
                    dict(long_name='mask on PSI-points')),
                mask_u=(
                    ['eta_u', 'xi_u'], self.hgrid.mask_u,
                    dict(long_name='mask on U-points')),
                mask_v=(
                    ['eta_v', 'xi_v'], self.hgrid.mask_v,
                    dict(long_name='mask on V-points')),
                ),
            coords=dict(
                theta_s=self.vgrid.theta_s,
                theta_b=self.vgrid.theta_b,
                Tcline=self.vgrid.Tcline,
                hc=self.vgrid.hc,
                Vtransform=self.vgrid.Vtrans,
                Vstretching=self.vgrid.Vstretch,
                xl=self.hgrid.xl,
                el=self.hgrid.el,
                xi_rho=range(xn),
                eta_rho=range(en),
                xi_vert=range(xn+1),
                eta_vert=range(en+1),
                xi_psi=range(xn-1),
                eta_psi=range(en-1),
                xi_u=range(xn-1),
                eta_u=range(en),
                xi_v=range(xn),
                eta_v=range(en-1),
                s_rho=self.vgrid.s_r,
                s_w=self.vgrid.s_w,
                x_vert=(
                    ['eta_vert', 'xi_vert'], self.hgrid.x_vert,
                    dict(long_name='x location of cell verticies',
                         units='meter')),
                y_vert=(
                    ['eta_vert', 'xi_vert'], self.hgrid.y_vert,
                    dict(long_name='y location of cell verticies',
                         units='meter')),
                x_rho=(
                    ['eta_rho', 'xi_rho'], self.hgrid.x_rho,
                    dict(long_name='x location of RHO-points',
                         units='meter')),
                y_rho=(
                    ['eta_rho', 'xi_rho'], self.hgrid.y_rho,
                    dict(long_name='y location of RHO-points',
                         units='meter')),
                x_psi=(
                    ['eta_psi', 'xi_psi'], self.hgrid.x_psi,
                    dict(long_name='x location of PSI-points',
                         units='meter')),
                y_psi=(
                    ['eta_psi', 'xi_psi'], self.hgrid.y_psi,
                    dict(long_name='y location of PSI-points',
                         units='meter')),
                x_u=(
                    ['eta_u', 'xi_u'], self.hgrid.x_u,
                    dict(long_name='x location of U-points',
                         units='meter')),
                y_u=(
                    ['eta_u', 'xi_u'], self.hgrid.y_u,
                    dict(long_name='y location of U-points',
                         units='meter')),
                x_v=(
                    ['eta_v', 'xi_v'], self.hgrid.x_v,
                    dict(long_name='x location of V-points',
                         units='meter')),
                y_v=(
                    ['eta_v', 'xi_v'], self.hgrid.y_v,
                    dict(long_name='y location of V-points',
                         units='meter')),
                pm=(
                    ['eta_rho', 'xi_rho'], self.hgrid.pm,
                    dict(long_name='curvilinear coordinate metric ' +
                                   ' in XI',
                         units='meter-1')),
                pn=(
                    ['eta_rho', 'xi_rho'], self.hgrid.pn,
                    dict(long_name='curvilinear coordinate metric ' +
                                   ' in ETA',
                         units='meter-1')),
                Cs_r=(
                    ['s_rho'], self.vgrid.Cs_r,
                    dict(long_name='S-coordinate stretching ' +
                                   'curves at RHO-points')),
                Cs_w=(
                    ['s_w'], self.vgrid.Cs_w,
                    dict(long_name='S-coordinate stretching ' +
                                   'curves at W-points')),
                ),
            attrs=dict(
                Name=self.name,
                Description='ROMS grid',
                Author='pyroms.grid.xarray_write_ROMS_grd',
                spherical=self.hgrid.spherical,
                )
            )

        ds_grd.s_rho.attrs['long_name'] = 'S-coordinate at RHO-points'
        ds_grd.s_w.attrs['long_name'] = 'S-coordinate at W-points'

        if self.hgrid.spherical:
            ds_grd.attrs['proj'] = self.hgrid.proj.crs.to_string()
            ds_grd.coords['lon_vert'] = (
                ['eta_vert', 'xi_vert'], self.hgrid.lon_vert,
                dict(long_name='longitude of cell verticies',
                     units='degree_east'))
            ds_grd.coords['lat_vert'] = (
                ['eta_vert', 'xi_vert'], self.hgrid.lat_vert,
                dict(long_name='latitude of cell verticies',
                     units='degree_north'))
            ds_grd.coords['lon_rho'] = (
                ['eta_rho', 'xi_rho'], self.hgrid.lon_rho,
                dict(long_name='longitude of RHO-points',
                     units='degree_east'))
            ds_grd.coords['lat_rho'] = (
                ['eta_rho', 'xi_rho'], self.hgrid.lat_rho,
                dict(long_name='latitude of RHO-points',
                     units='degree_north'))
            ds_grd.coords['lon_psi'] = (
                ['eta_psi', 'xi_psi'], self.hgrid.lon_psi,
                dict(long_name='longitude of PSI-points',
                     units='degree_east'))
            ds_grd.coords['lat_psi'] = (
                ['eta_psi', 'xi_psi'], self.hgrid.lat_psi,
                dict(long_name='latitude of PSI-points',
                     units='degree_north'))
            ds_grd.coords['lon_u'] = (
                ['eta_u', 'xi_u'], self.hgrid.lon_u,
                dict(long_name='longitude of U-points',
                     units='degree_east'))
            ds_grd.coords['lat_u'] = (
                ['eta_u', 'xi_u'], self.hgrid.lat_u,
                dict(long_name='latitude of U-points',
                     units='degree_north'))
            ds_grd.coords['lon_v'] = (
                ['eta_v', 'xi_v'], self.hgrid.lon_v,
                dict(long_name='longitude of V-points',
                     units='degree_east'))
            ds_grd.coords['lat_v'] = (
                ['eta_v', 'xi_v'], self.hgrid.lat_v,
                dict(long_name='latitude of V-points',
                     units='degree_north'))

        if hasattr(self.vgrid, 'zice'):
            ds_grd['zice'] = (
                ['eta_rho', 'xi_rho'], self.vgrid.zice,
                dict(long_name='iceshelf depth at RHO-points',
                     units='meter'))
            if hasattr(self.hgrid, 'mask_is'):
                ds_grd['mask_is'] = (
                    ['eta_rho', 'xi_rho'], self.hgrid.mask_is,
                    dict(long_name='mask of iceshelf on RHO-points'))

        return ds_grd

    def get_position(self, **kwargs):
        sta_hgrd = self.hgrid.get_position(**kwargs)
        return StaGrid('Selected Track', sta_hgrd, self.vgrid)


class ROMSGridInfo:
    '''
    gridinfo = ROMSGridInfo(gridid, grid_file=None, hist_file=None)

    Return an object with grid information for gridid.

    There are two ways to define the grid information.  If grid_file and
        hist_file are not passed to the object when it is created, the
        information is retrieved from gridid.txt.

    To add new grid please edit your gridid.txt. You need to define
        an environment variable PYROMS_GRIDID_FILE pointing to your
        gridid.txt file. Just copy an existing grid and modify the
        definition accordingly to your case (Be carefull with
        space and blank line).

    If grid_file is the path to a ROMS grid file, and hist_file is the
        path to a ROMS history file, then the grid information will be
        read from those files.  Gridid can then be used to refer to this
        grid information so that the grid and history files do not be
        included in subsequent calls.
    '''

    def __init__(self, gridid, grid_file=None, hist_file=None):
        # first determine if the information for the gridid has already
        # been obtained.
        if gridid in gridid_dictionary:
            saved_self = gridid_dictionary[gridid]
            for attrib in list(saved_self.__dict__.keys()):
                setattr(self, attrib, getattr(saved_self, attrib))
        else:
            # nope, we need to get the information from gridid.txt or from
            # the grid and history files from the model
            self.id = gridid
            self._get_grid_info(grid_file, hist_file)

            # now save the data in the dictionary, so we don't need to
            # get it again
            gridid_dictionary[gridid] = self

    def _get_grid_info(self, grid_file, hist_file):

        # check if the grid_file and hist_files are both null; if so get
        # data from gridid.txt
        if grid_file is None and hist_file is None:
            # print 'CJMP> gridid not in dictionary, data will be retrieved
            # from gridid.txt'
            gridid_file = os.getenv("PYROMS_GRIDID_FILE")
            file_type = gridid_file.split('.')[-1]
            if file_type in ['yml']:
                fh = open(gridid_file, 'r')
                data = yaml.safe_load_all(fh)
                for di in data:
                    if self.id == di['id']:
                        self.name = di['name']
                        self.grdfile = di['grdfile']
                        self.N = di['N']
                        self.grdtype = di['grdtype']
                        if self.grdtype == 'roms':
                            self.Vtrans = di['Vtrans']
                            self.Vstretch = di['Vstretch']
                            self.theta_s = di['theta_s']
                            self.theta_b = di['theta_b']
                            self.Tcline = di['Tcline']
                        elif self.grdtype == 'z':
                            self.depth = np.array(di['depth'], dtype=float)
                        else:
                            raise ValueError(
                                'Unknown grid type. Please check your ' +
                                'gridid.txt file')
                    else:
                        raise ValueError(
                            'Unknown gridid. Please check your ' +
                            'gridid.txt file')
            else:
                data = open(gridid_file, 'r')
                lines = data.readlines()
                data.close()

                line_nb = 0
                info = []
                for line in lines:
                    s = line.split()
                    if s[0] == 'id':
                        if s[2] == self.id:
                            for ll in range(line_nb, line_nb+5):
                                s = lines[ll].split()
                                info.append(s[2])
                                line_nb = line_nb + 1
                            if info[4] == 'roms':
                                for ll in range(line_nb, line_nb+4):
                                    s = lines[ll].split()
                                    info.append(s[2])
                            if info[4] == 'z':
                                s = lines[line_nb].split()
                                info.append(s[3:-1])
                                while s[-1:] == ['\\']:
                                    line_nb = line_nb + 1
                                    s = lines[line_nb].split()
                                    info.append(s[:-1])
                    line_nb = line_nb + 1

                if info == []:
                    raise ValueError(
                        'Unknown gridid. Please check your gridid.txt file')
                if info[4] == 'roms':
                    self.name = info[1]
                    self.grdfile = info[2]
                    self.N = np.int(info[3])
                    self.grdtype = info[4]
                    self.Vtrans = np.int(info[5])
                    self.Vstretch = np.int(info[6])
                    self.theta_s = np.float(info[7])
                    self.theta_b = np.float(info[8])
                    self.Tcline = np.float(info[9])
                elif info[4] == 'z':
                    nline = len(info)
                    dep = info[5]
                    for line in range(6, nline):
                        dep = dep + info[line]
                    dep = np.array(dep, dtype=np.float)

                    self.name = info[1]
                    self.grdfile = info[2]
                    self.N = np.int(info[3])
                    self.grdtype = info[4]
                    self.depth = dep
                else:
                    raise ValueError(
                        'Unknown grid type. Please check your gridid.txt file')
        else:
            # lets get the grid information from the history and grid files
            assert ~isinstance(grid_file, type(None)), \
                'if specify history file, must specify grid file'
            if hist_file is None:
                hist_file = grid_file

            # open history file and get necessary grid information from it.
            hist = nc.Dataset(hist_file, 'r')

            # put data into ROMSGridInfo object
            self.name = grid_file.split('/')[-1].split('.nc')[0]
            self.grdfile = grid_file
            self.N = len(hist.dimensions['s_rho'])
            self.grdtype = 'roms'

            try:
                self.Vtrans = np.float(hist.variables['Vtransform'][:])
                self.Vstretch = np.float(hist.variables['Vstretching'][:])
            except KeyError:
                print('variable Vtransform/Vstretching not found in ' +
                      'history file. Defaulting to Vtrans=2 and ' +
                      'Vstretch=4')
                self.Vtrans = 2
                self.Vstretch = 4
            self.theta_s = np.float(hist.variables['theta_s'][:])
            self.theta_b = np.float(hist.variables['theta_b'][:])
            self.Tcline = np.float(hist.variables['Tcline'][:])


def get_ROMS_hgrid(gridid):
    """
    hgrid = get_ROMS_hgrid(gridid)

    Load ROMS horizontal grid object
    """

    gridinfo = ROMSGridInfo(gridid)
    grdfile = gridinfo.grdfile

    fh = io.Dataset(grdfile)
    try:
        fh.set_always_mask(False)
    except Exception:
        pass

    # Check for cartesian or geographical grid
    spherical = fh.variables['spherical'][0].item()
    if spherical == 0 or spherical == 'F':
        spherical = False
    else:
        spherical = True

    # Get horizontal grid
    if not spherical:
        # cartesian grid
        print('Load cartesian grid from file')
        if 'x_vert' in list(fh.variables.keys()) and \
           'y_vert' in list(fh.variables.keys()):
            x_vert = fh.variables['x_vert'][:]
            y_vert = fh.variables['y_vert'][:]
        elif 'x_rho' in list(fh.variables.keys()) and \
             'y_rho' in list(fh.variables.keys()) and \
             'pm' in list(fh.variables.keys()) and \
             'pn' in list(fh.variables.keys()):
            x_rho = fh.variables['x_rho'][:]
            y_rho = fh.variables['y_rho'][:]
            pm = fh.variables['pm'][:]
            pn = fh.variables['pn'][:]
            try:
                angle = fh.variables['angle'][:]
            except Exception:
                angle = np.zeros(x_rho.shape)
            # compute verts from rho point, pm, pn, angle
            x_vert, y_vert = rho_to_vert(x_rho, y_rho, pm, pn, angle)
        else:
            raise ValueError(
                'NetCDF file must contain x_vert and y_vert ' +
                'or x_rho, y_rho, pm, pn and angle for a cartesian grid')

        if 'x_rho' in list(fh.variables.keys()) and \
           'y_rho' in list(fh.variables.keys()) and \
           'x_u' in list(fh.variables.keys()) and \
           'y_u' in list(fh.variables.keys()) and \
           'x_v' in list(fh.variables.keys()) and \
           'y_v' in list(fh.variables.keys()) and \
           'x_psi' in list(fh.variables.keys()) and \
           'y_psi' in list(fh.variables.keys()):
            x_rho = fh.variables['x_rho'][:]
            y_rho = fh.variables['y_rho'][:]
            x_u = fh.variables['x_u'][:]
            y_u = fh.variables['y_u'][:]
            x_v = fh.variables['x_v'][:]
            y_v = fh.variables['y_v'][:]
            x_psi = fh.variables['x_psi'][:]
            y_psi = fh.variables['y_psi'][:]
        else:
            x_rho = None
            y_rho = None
            x_u = None
            y_u = None
            x_v = None
            y_v = None
            x_psi = None
            y_psi = None

        if 'pm' in list(fh.variables.keys()) and \
           'pn' in list(fh.variables.keys()):
            pm = fh.variables['pm'][:]
            dx = 1. / pm
            pn = fh.variables['pn'][:]
            dy = 1. / pn
        else:
            dx = None
            dy = None

        if 'dndx' in list(fh.variables.keys()) and \
           'dmde' in list(fh.variables.keys()):
            dndx = fh.variables['dndx'][:]
            dmde = fh.variables['dmde'][:]
        else:
            dndx = None
            dmde = None

        if 'angle' in list(fh.variables.keys()):
            angle = fh.variables['angle'][:]
        else:
            angle = None

        # Get cartesian grid
        hgrd = CGrid(x_vert, y_vert, x_rho=x_rho, y_rho=y_rho,
                     x_u=x_u, y_u=y_u, x_v=x_v, y_v=y_v,
                     x_psi=x_psi, y_psi=y_psi, dx=dx, dy=dy,
                     dndx=dndx, dmde=dmde, angle_rho=angle)

    else:
        # geographical grid
        print('Load geographical grid from file')
        if 'lon_vert' in list(fh.variables.keys()) and \
           'lat_vert' in list(fh.variables.keys()):
            lon_vert = fh.variables['lon_vert'][:]
            lat_vert = fh.variables['lat_vert'][:]
        elif 'lon_rho' in list(fh.variables.keys()) and \
             'lat_rho' in list(fh.variables.keys()) and \
             'lon_psi' in list(fh.variables.keys()) and \
             'lat_psi' in list(fh.variables.keys()):
            lon_rho = fh.variables['lon_rho'][:]
            lat_rho = fh.variables['lat_rho'][:]
            lon_psi = fh.variables['lon_psi'][:]
            lat_psi = fh.variables['lat_psi'][:]
            # compute verts from rho and psi point
            lon_vert, lat_vert = rho_to_vert_geo(
                lon_rho, lat_rho, lon_psi, lat_psi)
        else:
            raise ValueError('NetCDF file must contain lon_vert and lat_vert \
                or lon_rho, lat_rho, lon_psi, lat_psi for a geographical grid')

        if 'lon_rho' in list(fh.variables.keys()) and \
           'lat_rho' in list(fh.variables.keys()) and \
           'lon_u' in list(fh.variables.keys()) and \
           'lat_u' in list(fh.variables.keys()) and \
           'lon_v' in list(fh.variables.keys()) and \
           'lat_v' in list(fh.variables.keys()) and \
           'lon_psi' in list(fh.variables.keys()) and \
           'lat_psi' in list(fh.variables.keys()):
            lon_rho = fh.variables['lon_rho'][:]
            lat_rho = fh.variables['lat_rho'][:]
            lon_u = fh.variables['lon_u'][:]
            lat_u = fh.variables['lat_u'][:]
            lon_v = fh.variables['lon_v'][:]
            lat_v = fh.variables['lat_v'][:]
            lon_psi = fh.variables['lon_psi'][:]
            lat_psi = fh.variables['lat_psi'][:]
        else:
            lon_rho = None
            lat_rho = None
            lon_u = None
            lat_u = None
            lon_v = None
            lat_v = None
            lon_psi = None
            lat_psi = None

        if 'pm' in list(fh.variables.keys()) and \
           'pn' in list(fh.variables.keys()):
            pm = fh.variables['pm'][:]
            dx = 1. / pm
            pn = fh.variables['pn'][:]
            dy = 1. / pn
        else:
            dx = None
            dy = None

        if 'dndx' in list(fh.variables.keys()) and \
                'dmde' in list(fh.variables.keys()):
            dndx = fh.variables['dndx'][:]
            dmde = fh.variables['dmde'][:]
        else:
            dndx = None
            dmde = None

        if 'angle' in list(fh.variables.keys()):
            angle = fh.variables['angle'][:]
        else:
            angle = None

        if hasattr(fh, 'proj4_init'):
            proj = pyproj.Proj(fh.proj4_init)
        else:
            if (lon_vert is not None) and (lat_vert is not None):
                i, j = lon_vert.shape
                lon0, lat0 = \
                    lon_vert[int(i/2), int[j/2]], lat_vert[int(i/2), int(j/2)]
            else:
                i, j = lon_rho.shape
                lon0, lat0 = \
                    lon_rho[int(i/2), int[j/2]], lat_rho[int(i/2), int[j/2]]
            proj = pyproj.Proj(proj='stere', lat_0=lat0, lon_0=lon0)

        # Get geographical grid
        hgrd = CGridGeo(lon_vert, lat_vert, proj,
                        lon_rho=lon_rho, lat_rho=lat_rho,
                        lon_u=lon_u, lat_u=lat_u, lon_v=lon_v, lat_v=lat_v,
                        lon_psi=lon_psi, lat_psi=lat_psi, dx=dx, dy=dy,
                        dndx=dndx, dmde=dmde, angle_rho=angle)

    # load the mask
    try:
        hgrd.mask_rho = fh.variables['mask_rho'][:].astype(np.int32)
    except KeyError:
        hgrd.mask_rho = np.ones(hgrd.x_rho.shape, dtype=np.int32)
    if 'mask_is' in fh.variables.keys():
        hgrd.mask_is = fh.variables['mask_is'][:].astype(np.int32)
    elif 'mask_iceshelf' in fh.variables.keys():
        hgrd.mask_is = fh.variables['mask_iceshelf'][:].astype(np.int32)

    fh.close()

    return hgrd


def get_ROMS_vgrid(gridid, zeta=None):
    """
    vgrid = get_ROMS_vgrid(gridid)

    Load ROMS vertical grid object. vgrid is a SCoord or a ZCoord object,
        depending on gridid.grdtype.

    vgrid.z_r and vgrid.z_w (vgrid.z for a z_coordinate object)
        can be indexed in order to retreive the actual depths. The
        free surface time serie zeta can be provided as an optional
        argument. Note that the values of zeta are not calculated
        until z is indexed, so a netCDF variable for zeta may be passed,
        even if the file is large, as only the values that are required
        will be retrieved from the file.
    """

    gridinfo = ROMSGridInfo(gridid)
    grdfile = gridinfo.grdfile

    fh = io.Dataset(grdfile)

    # Get vertical grid
    try:
        h = fh.variables['h'][:]
    except Exception:
        raise ValueError('NetCDF file must contain the bathymetry h')

    if 'zice' in fh.variables.keys():
        zice = fh.variables['zice'][:]
        if 'ziceraw' in fh.variables.keys():
            ziceraw = fh.variables['ziceraw'][:]
        else:
            ziceraw = None
    else:
        zice = None
        ziceraw = None

    if 'hraw' in fh.variables.keys():
        hraw = fh.variables['hraw'][:]
    else:
        hraw = None

    if gridinfo.grdtype == 'roms':
        Vtrans = gridinfo.Vtrans
        Vstretch = gridinfo.Vstretch
        theta_b = gridinfo.theta_b
        theta_s = gridinfo.theta_s
        Tcline = gridinfo.Tcline
        N = gridinfo.N
        vgrid = SCoord(h, theta_b, theta_s, Tcline, N, Vtrans, Vstretch,
                       hraw=hraw, zeta=zeta, zice=zice)

    elif gridinfo.grdtype == 'z':
        N = gridinfo.N
        depth = gridinfo.depth
        vgrid = ZCoord(h, depth, N)

    else:
        raise ValueError('Unknown grid type')

    vgrid.ziceraw = ziceraw

    fh.close()

    return vgrid


def get_ROMS_grid(grid_file, hist_file=None, gridid=None, zeta=None,
                  auto_mask=True):
    """
    grd = get_ROMS_grid(grid_file, hist_file=None, gridid=None, zeta=None,
                        auto_mask=True)

    Load ROMS grid object.

    gridid is a string with the name of the grid in it. If hist_file and
        grid_file are not passed into the function, or are set to None,
        then gridid is used to get the grid data from the gridid.txt file.

    if hist_file and grid_file are given, and they are the file paths to
        a ROMS history file and grid file respectively, the grid information
        will be extracted from those files, and gridid will be used to name
        that grid for the rest of the python session.

    grd.vgrid is a SCoord or a ZCoord object, depending on gridid.grdtype.
        grd.vgrid.z_r and grd.vgrid.z_w (grd.vgrid.z for a z_coordinate object)
        can be indexed in order to retreive the actual depths. The free surface
        time serie zeta can be provided as an optional argument. Note that the
        values of zeta are not calculated until z is indexed, so a netCDF
        variable for zeta may be passed, even if the file is large, as only the
        values that are required will be retrieved from the file.
    """

    # in this first call to ROMSGridInfo, we pass in the history file
    # and gridfile info.  If hist_file and grid_file are defined, the
    # grid info will be extracted from those files and will able to be
    # accessed later by gridid
    gridinfo = ROMSGridInfo(gridid, hist_file=hist_file, grid_file=grid_file)
    name = gridinfo.name

    # we need not pass in hist_file and grid_file here, because the
    # gridinfo file will already have been initialized by the call to
    # ROMSGridInfo above.
    hgrid = get_ROMS_hgrid(gridid)
    vgrid = get_ROMS_vgrid(gridid, zeta=zeta)
    if auto_mask:
        vgrid.h = np.ma.masked_where(hgrid.mask_rho == 0, vgrid.h)
        if hasattr(vgrid, 'zice'):
            vgrid.zice = np.ma.masked_where(hgrid.mask_rho == 0, vgrid.zice)

    # Get ROMS grid
    return ROMSGrid(name, hgrid, vgrid)


def write_ROMS_grid(grd, filename='roms_grd.nc'):
    """
    write_ROMS_grid(grd, filename)

    Write ROMS_CGrid class on a NetCDF file.
    """

    Mm, Lm = grd.hgrid.x_rho.shape

    # Write ROMS grid to file
    fh = nc.Dataset(filename, 'w')
    fh.Description = 'ROMS grid'
    fh.Author = 'pyroms.grid.write_grd'
    fh.Created = datetime.now().isoformat()
    fh.type = 'ROMS grid file'
    if isinstance(grd.hgrid, CGridGeo):
        if isinstance(grd.hgrid.proj, pyproj.Proj):
            fh.proj4_init = grd.hgrid.proj.crs.to_wkt()
        elif hasattr(grd.hgrid.proj, 'proj4string'):
            fh.proj4_init = grd.hgrid.proj.proj4string
        else:
            fh.proj4_init = ''

    fh.createDimension('xi_rho', Lm)
    fh.createDimension('xi_u', Lm-1)
    fh.createDimension('xi_v', Lm)
    fh.createDimension('xi_psi', Lm-1)

    fh.createDimension('eta_rho', Mm)
    fh.createDimension('eta_u', Mm)
    fh.createDimension('eta_v', Mm-1)
    fh.createDimension('eta_psi', Mm-1)

    fh.createDimension('xi_vert', Lm+1)
    fh.createDimension('eta_vert', Mm+1)

    fh.createDimension('bath', None)

    if hasattr(grd.vgrid, 's_r'):
        if grd.vgrid.s_r is not None:
            fh.createDimension('s_rho', grd.vgrid.N)
            fh.createDimension('s_w', grd.vgrid.N+1)

    if hasattr(grd.vgrid, 's_r'):
        if grd.vgrid.s_r is not None:
            io.nc_write_var(fh, grd.vgrid.theta_s, 'theta_s', (),
                            'S-coordinate surface control parameter')
            io.nc_write_var(fh, grd.vgrid.theta_b, 'theta_b', (),
                            'S-coordinate bottom control parameter')
            io.nc_write_var(fh, grd.vgrid.Tcline, 'Tcline', (),
                            'S-coordinate surface/bottom layer width', 'meter')
            io.nc_write_var(fh, grd.vgrid.hc, 'hc', (),
                            'S-coordinate parameter, critical depth', 'meter')
            io.nc_write_var(fh, grd.vgrid.s_r, 's_rho', ('s_rho'),
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

    io.nc_write_var(fh, grd.vgrid.h, 'h', ('eta_rho', 'xi_rho'),
                    'bathymetry at RHO-points', 'meter')
    if hasattr(grd.vgrid, 'zice') is True:
        io.nc_write_var(fh, grd.vgrid.zice, 'zice', ('eta_rho', 'xi_rho'),
                        'iceshelf depth at RHO-points', 'meter')
        if hasattr(grd.vgrid, 'ziceraw') is True:
            io.nc_write_var(fh, grd.vgrid.ziceraw, 'ziceraw',
                            ('eta_rho', 'xi_rho'),
                            'raw iceshelf depth at RHO-points', 'meter')

    # ensure that we have a bath dependancy for hraw
    if grd.vgrid.hraw is None:
        grd.vgrid.hraw = grd.vgrid.h
    if len(grd.vgrid.hraw.shape) == 2:
        hraw = np.zeros(
            (1, grd.vgrid.hraw.shape[0], grd.vgrid.hraw.shape[1]))
        hraw[0, :] = grd.vgrid.hraw
    else:
        hraw = grd.vgrid.hraw
    io.nc_write_var(fh, hraw, 'hraw', ('bath', 'eta_rho', 'xi_rho'),
                    'raw bathymetry at RHO-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.f, 'f', ('eta_rho', 'xi_rho'),
                    'Coriolis parameter at RHO-points', 'second-1')
    io.nc_write_var(fh, 1./grd.hgrid.dx, 'pm', ('eta_rho', 'xi_rho'),
                    'curvilinear coordinate metric in XI', 'meter-1')
    io.nc_write_var(fh, 1./grd.hgrid.dy, 'pn', ('eta_rho', 'xi_rho'),
                    'curvilinear coordinate metric in ETA', 'meter-1')
    io.nc_write_var(fh, grd.hgrid.dmde, 'dmde', ('eta_rho', 'xi_rho'),
                    'XI derivative of inverse metric factor pn', 'meter')
    io.nc_write_var(fh, grd.hgrid.dndx, 'dndx', ('eta_rho', 'xi_rho'),
                    'ETA derivative of inverse metric factor pm', 'meter')
    io.nc_write_var(fh, grd.hgrid.xl, 'xl', (),
                    'domain length in the XI-direction', 'meter')
    io.nc_write_var(fh, grd.hgrid.el, 'el', (),
                    'domain length in the ETA-direction', 'meter')

    io.nc_write_var(fh, grd.hgrid.x_rho, 'x_rho', ('eta_rho', 'xi_rho'),
                    'x location of RHO-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.y_rho, 'y_rho', ('eta_rho', 'xi_rho'),
                    'y location of RHO-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.x_u, 'x_u', ('eta_u', 'xi_u'),
                    'x location of U-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.y_u, 'y_u', ('eta_u', 'xi_u'),
                    'y location of U-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.x_v, 'x_v', ('eta_v', 'xi_v'),
                    'x location of V-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.y_v, 'y_v', ('eta_v', 'xi_v'),
                    'y location of V-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.x_psi, 'x_psi', ('eta_psi', 'xi_psi'),
                    'x location of PSI-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.y_psi, 'y_psi', ('eta_psi', 'xi_psi'),
                    'y location of PSI-points', 'meter')
    io.nc_write_var(fh, grd.hgrid.x_vert, 'x_vert', ('eta_vert', 'xi_vert'),
                    'x location of cell verticies', 'meter')
    io.nc_write_var(fh, grd.hgrid.y_vert, 'y_vert', ('eta_vert', 'xi_vert'),
                    'y location of cell verticies', 'meter')

    if hasattr(grd.hgrid, 'lon_rho'):
        io.nc_write_var(fh, grd.hgrid.lon_rho, 'lon_rho',
                        ('eta_rho', 'xi_rho'),
                        'longitude of RHO-points', 'degree_east')
        io.nc_write_var(fh, grd.hgrid.lat_rho, 'lat_rho',
                        ('eta_rho', 'xi_rho'),
                        'latitude of RHO-points', 'degree_north')
        io.nc_write_var(fh, grd.hgrid.lon_psi, 'lon_psi',
                        ('eta_psi', 'xi_psi'),
                        'longitude of PSI-points', 'degree_east')
        io.nc_write_var(fh, grd.hgrid.lat_psi, 'lat_psi',
                        ('eta_psi', 'xi_psi'),
                        'latitude of PSI-points', 'degree_north')
        io.nc_write_var(fh, grd.hgrid.lon_vert, 'lon_vert',
                        ('eta_vert', 'xi_vert'),
                        'longitude of cell verticies', 'degree_east')
        io.nc_write_var(fh, grd.hgrid.lat_vert, 'lat_vert',
                        ('eta_vert', 'xi_vert'),
                        'latitude of cell verticies', 'degree_north')
        io.nc_write_var(fh, grd.hgrid.lon_u, 'lon_u', ('eta_u', 'xi_u'),
                        'longitude of U-points', 'degree_east')
        io.nc_write_var(fh, grd.hgrid.lat_u, 'lat_u', ('eta_u', 'xi_u'),
                        'latitude of U-points', 'degree_north')
        io.nc_write_var(fh, grd.hgrid.lon_v, 'lon_v', ('eta_v', 'xi_v'),
                        'longitude of V-points', 'degree_east')
        io.nc_write_var(fh, grd.hgrid.lat_v, 'lat_v', ('eta_v', 'xi_v'),
                        'latitude of V-points', 'degree_north')

    fh.createVariable('spherical', 'i')
    fh.variables['spherical'].long_name = 'Grid type logical switch'
    if grd.hgrid.spherical:
        fh.variables['spherical'][:] = 1
    else:
        fh.variables['spherical'][:] = 0
    print(' ... wrote ', 'spherical')

    io.nc_write_var(fh, grd.hgrid.angle_rho, 'angle', ('eta_rho', 'xi_rho'),
                    'angle between XI-axis and EAST', 'radians')
    io.nc_write_var(fh, grd.hgrid.mask_rho.astype(float), 'mask_rho',
                    ('eta_rho', 'xi_rho'), 'mask on RHO-points')
    io.nc_write_var(fh, grd.hgrid.mask_u.astype(float), 'mask_u',
                    ('eta_u', 'xi_u'), 'mask on U-points')
    io.nc_write_var(fh, grd.hgrid.mask_v.astype(float), 'mask_v',
                    ('eta_v', 'xi_v'), 'mask on V-points')
    io.nc_write_var(fh, grd.hgrid.mask_psi.astype(float), 'mask_psi',
                    ('eta_psi', 'xi_psi'), 'mask on psi-points')
    if hasattr(grd.hgrid, 'mask_is'):
        io.nc_write_var(fh, grd.hgrid.mask_is.astype(float),
                        'mask_is', ('eta_rho', 'xi_rho'),
                        'mask of iceshelf on RHO-points')

    fh.close()


def print_ROMSGridInfo(gridid):
    """
    print_ROMSGridInfo(gridid)

    return the grid information for gridid
    """

    gridinfo = ROMSGridInfo(gridid)

    print(' ')
    print('grid information for gridid ', gridinfo.id, ':')
    print(' ')
    print('grid name : ', gridinfo.name)
    print('grid file path : ', gridinfo.grdfile)
    print('number of vertical level : ', gridinfo.N)
    print('grid type : ', gridinfo.grdtype)
    if gridinfo.grdtype == 'roms':
        print('theta_s = ', gridinfo.theta_s)
        print('theta_b = ', gridinfo.theta_b)
        print('Tcline  = ', gridinfo.Tcline)
        # print 'hc      = ', gridinfo.hc
    elif gridinfo.grdtype == 'z':
        print('depth = ', gridinfo.depth)


def list_ROMS_gridid():
    """
    list_ROMS_gridid()

    return the list of the defined gridid
    """

    gridid_file = os.getenv("PYROMS_GRIDID_FILE")
    data = open(gridid_file, 'r')
    lines = data.readlines()
    data.close()

    gridid_list = []
    for line in lines:
        s = line.split()
        if s[0] == 'id':
            gridid_list.append(s[2])

    print('List of defined gridid : ', gridid_list)
