import functools
import numpy as np
import pyproj
import xarray as xr
from xarray import DataArray
from xgcm import Grid
import matplotlib.pyplot as plt
from . import grid, hgrid, vgrid


def _find_hpos(obj) -> str:
    """
    Pass in a Dataset/DataArray to find the vertex horizontal coordinate
    position (rho/u/v) of the data to be worked with. Searth order is
    rho/u/v/psi/vert.

    pos = _find_hpos(obj)
    """

    if 'eta_rho' in obj.dims or 'xi_rho' in obj.dims:
        pos = '_rho'
    elif 'eta_u' in obj.dims or 'xi_u' in obj.dims:
        pos = '_u'
    elif 'eta_v' in obj.dims or 'xi_v' in obj.dims:
        pos = '_v'
    elif 'eta_psi' in obj.dims or 'xi_psi' in obj.dims:
        pos = '_psi'
    elif 'eta_vert' in obj.dims or 'xi_vert' in obj.dims:
        pos = '_vert'
    else:
        pos = ''
    return pos


def _find_vpos(obj) -> str:
    """
    Pass in a Dataset/DataArray to find the vertex vertical coordinate
    position (r/w) of the data to be worked with.
    If obj is a Dataset, 'r' will be searched first, and then w.

    pos = _find_vpos(obj)
    """

    if 's_rho' in obj.dims:
        pos = '_r'
    elif 's_w' in obj.dims:
        pos = '_w'
    else:
        pos = ''
    return pos


def _set_hzz(obj, pos, clip_z=True) -> (DataArray, DataArray, DataArray):
    """
    Find the coordinate (rho/u/v) associated with the DataArray to
    calculate depth, and prepare data for the calculation.

    h, zice, zeta = _set_hzz(obj, pos, clip_z=True)

    Input:
        obj  - Xarray Dataset or DataArray
        pos  - name or the coordinate (rho/u/v)
    Outputs:
        h    - bottom bathymetry
        zice - ice shelf draft depth
        zeta - sea surface
    """

    # Fetch h/zice/zeta.
    h = obj['h' + pos]
    h = h.where(~h.isnull(), 0)
    if 'zice' + pos in obj.coords:
        zice = obj['zice' + pos]
        zice = zice.where(~zice.isnull(), 0)
    else:
        zice = xr.zeros_like(h)
    if 'zeta' in obj.coords:
        zeta = obj['zeta' + pos]
        zeta = zeta.where(~zeta.isnull(), 0)
        if 'ocean_time' in zeta.dims and not clip_z:
            zeta = zeta.mean(dim='ocean_time')
    else:
        zeta = xr.zeros_like(h)
    return h, zice, zeta


def _set_z(h: DataArray, zice: DataArray, zeta: DataArray,
           s: DataArray, Cs: DataArray,
           hc: float, Vtransform: int) -> DataArray:
    """
    Calculate grid z-coord depth given water depth (h), iceshelf depth (zice),
    sea surface (zeta), and vertical grid transformation parameters.

    z = _set_z(h, zice, zeta, s, Cs, hc, Vtransform)

    Inputs:
        h, zice, zeta         - bathymetry extracted by set_hzz
        s, Cs, hc, Vtransform - ROMS grid transformation parameters
    Output:
        z - depth of rho/w points
    """
    if Vtransform == 1:
        z0 = hc*s + (h-zice-hc)*Cs
        z = zeta*(1.0+z0/(h-zice)) + z0 - zice
    elif Vtransform == 2:
        z0 = (hc*s + (h-zice)*Cs) / (hc+h-zice)
        z = zeta + (zeta+h-zice)*z0 - zice

    # Transpose dims in z to match DataArray
    if 'ocean_time' in zeta.dims:
        dims = ('ocean_time', ) + s.dims + zeta.dims[1:]
    else:
        dims = s.dims + zeta.dims
    z = z.transpose(*dims)
    return z


def _calc_z(obj, vpos: str, hpos: str) -> DataArray:
    """
    Wrapper function for _set_z.

    z = _calc_z(vpos, hpos)
    """
    if vpos is None:
        return
    elif vpos == '_r':
        s = 's_rho'
    elif vpos == '_w':
        s = 's_w'
    h, zice, zeta = _set_hzz(obj, hpos)
    z = _set_z(h, zice, zeta,
               obj[s], obj['Cs' + vpos],
               obj.hc, obj.Vtransform)
    z.attrs = dict(
        long_name='z' + vpos + ' at ' + hpos[1:].upper() + '-points',
        units='meter')
    return z


def _zidx(zr: DataArray, zw: DataArray, z: float) -> DataArray:
    """
    Find the index where depth is z. Note that the returned value has both
    integer part and fractional part.

    zidx = _zidx(zr, zw, z)
    """
    zw = zw + abs(z)
    zint = zw.isel(s_w=slice(1, None)).data * \
        zw.isel(s_w=slice(None, -1)).data
    zint = DataArray(zint, dims=zr.dims, coords=zr.coords)
    zint = zint.where(zint <= 0, 1e37)
    zint = zint.argmin(dim='s_rho')

    z0 = zw.isel(s_w=zint)
    z1 = zw.isel(s_w=zint+1)
    zfrac = -z0/(z1-z0)
    zidx = zint + zfrac
    return zidx


class ROMSAccessor:
    """
    This is the basic ROMS accessor for both Xarray Dataset and DataArray.
    """

    def __init__(self, obj):
        self._obj = obj

    def set_locs(self, lon, lat, time=None):
        """
        Put lon/lat and x/y coords in a DataArray given lon, lat as
        tuple/list/ndarray.

        ds_locs = roms.set_locs(lon, lat, time=None)
        Inputs:
            lon, lat - 1-D list of longitude/latitude
            time (optional) - 1-D list of time
        Output:
            ds_locs - interpolation coordinates.
        """

        lon, lat = np.asarray(lon), np.asarray(lat)
        assert len(lon) == len(lat), 'lon/lat must have the same length.'

        # Pass in geographic information to a DataArray
        ds_locs = xr.Dataset()
        ds_locs['lon'] = DataArray(lon, dims=('track'))
        ds_locs['lat'] = DataArray(lat, dims=('track'))
        proj = pyproj.Proj(self._obj.attrs['proj4_init'])
        x, y = proj(ds_locs['lon'], ds_locs['lat'])
        ds_locs['x'] = DataArray(x, dims=('track'))
        ds_locs['y'] = DataArray(y, dims=('track'))

        if time is not None:
            time = np.asarray(time)
            if len(time) == len(lon):
                ds_locs['ocean_time'] = DataArray(time, dims=('track'))
            else:
                ds_locs['ocean_time'] = DataArray(time, dims=('ocean_time'))

        # Assign coordiantes and calculate distance from lon[0]/lat[0]
        dis = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]).cumsum()
        dis = np.concatenate((np.array([0]), dis))
        ds_locs = ds_locs.assign_coords(track=np.arange(len(lon)))
        ds_locs = ds_locs.assign_coords(distance=('track', dis))
        return ds_locs

    def longitude_wrap(self):
        """
        Use this function to switch longitude range between [-180 180] and
        [0 360].

        ds.longitude_wrap()
        """
        for lon in ['lon_rho', 'lon_psi', 'lon_u', 'lon_v', 'lon_vert']:
            if lon in self._obj.coords:
                attrs = self._obj[lon].attrs
                if self._obj.coords[lon].max() > 180.:
                    self._obj.coords[lon] = (self._obj[lon]+180) % 360 - 180
                    self._obj[lon].attrs = attrs
                elif self._obj.coords[lon].min() < 0.:
                    self._obj.coords[lon] = self._obj[lon] % 360
                    self._obj[lon].attrs = attrs

    @property
    def center(self):
        """
        A test function. Return the geographic center point of this dataset
        """
        for pos in ['_vert', '_rho', '_psi', '_u', '_v']:
            if 'x' + pos in self._obj and 'y' + pos in self._obj:
                if self._obj._spherical:
                    return self._obj._grid.hgrid.proj(
                        self._obj['x' + pos].mean(),
                        self._obj['y' + pos].mean(),
                        inverse=True)
                else:
                    return self._obj['x' + pos].mean(), \
                           self._obj['y' + pos].mean()


@xr.register_dataset_accessor("roms")
class ROMSDatasetAccessor(ROMSAccessor):

    def __init__(self, obj):
        """
        Pass in a Dataset object for ROMSDatasetAccessor.
        """
        super().__init__(obj)
        dims = obj.dims

        self._hpos = []
        if 'eta_rho' in dims or 'xi_rho' in dims:
            self._hpos.append('_rho')
        if 'eta_psi' in dims or 'xi_psi' in dims:
            self._hpos.append('_psi')
        if 'eta_u' in dims or 'xi_u' in dims:
            self._hpos.append('_u')
        if 'eta_v' in dims or 'xi_v' in dims:
            self._hpos.append('_v')
        if 'eta_vert' in dims or 'xi_vert' in dims:
            self._hpos.append('_vert')
        if len(self._hpos) == 0:
            self._hpos = ['']

        self._vpos = []
        if 's_rho' in obj.dims:
            self._vpos.append('_r')
        if 's_w' in obj.dims:
            self._vpos.append('_w')
        if len(self._vpos) == 0:
            self._vpos = ['']

    def interp(self, lon, lat, time=None):
        """
        Horizontal interpolation method for ROMS Dataset.

        ds = roms.interp(lon, lat, time=None)
        Inputs:
            lon, lat - 1-D list of longitude/latitude
            time (optional) - 1-D list of time
        Output:
            ds - interpolated Dataset.
        """
        ds_locs = self.set_locs(lon, lat, time)

        # Calculate disctance array and find the integer part of coordiantes.
        xdis, ydis = self.x - ds_locs['x'], self.y - ds_locs['y']
        dis_min = np.hypot(xdis, ydis).argmin(dim=('eta_rho', 'xi_rho'))

        # eta0, xi0 are the coords of left bottom corner.
        etav, xiv = dis_min['eta_rho'], dis_min['xi_rho']
        eta0, xi0 = self._obj.eta_vert.data[0], self._obj.xi_vert.data[0]

        # Calculate the fractional part of coordinates and add to integer part.
        eta_loc, xi_loc = [], []
        xv, yv = self._obj.x_vert.data, self._obj.y_vert.data
        for ei, xi, xx, yy in \
                zip(etav.data, xiv.data, ds_locs.x.data, ds_locs.y.data):
            ivec = np.array([xv[ei+1, xi]-xv[ei, xi], yv[ei+1, xi]-yv[ei, xi]])
            jvec = np.array([xv[ei, xi+1]-xv[ei, xi], yv[ei, xi+1]-yv[ei, xi]])
            c = np.array([xx-xv[ei, xi], yy-yv[ei, xi]])
            efrac = np.dot(ivec, c)/(np.dot(ivec, ivec))
            xfrac = np.dot(jvec, c)/(np.dot(jvec, jvec))

            eta_loc.append(ei + eta0 + efrac)
            xi_loc.append(xi + xi0 + xfrac)

        ds_locs['eta'] = DataArray(eta_loc, dims=('track'))
        ds_locs['xi'] = DataArray(xi_loc, dims=('track'))

        # Perform interpolation using Xarray's interp method
        interp_coords = {}
        for pos in self._hpos:
            interp_coords['eta' + pos] = ds_locs.eta
            interp_coords['xi' + pos] = ds_locs.xi
            if time is not None:
                interp_coords['ocean_time'] = ds_locs.ocean_time
        ds = self._obj.interp(interp_coords)

        # Also update horizontal coordinates
        for var in ['lat', 'lon', 'x', 'y', 'mask']:
            ds.coords[var] = ds[var + '_rho']
        ds.coords['eta'], ds.coords['xi'] = ds_locs.eta, ds_locs.xi

        # Clean up coordiantes with suffix rho/u/v/psi/vert.
        drop_coords = []
        for coord in ds.coords:
            if ('_rho' in coord or '_u' in coord or '_v' in coord or
                '_psi' in coord or '_vert' in coord) and \
               coord != 's_rho':
                drop_coords.append(coord)
        for coord in drop_coords:
            ds.__delitem__(coord)
        return ds

    def transform(self, z):
        """
        Vertical coordinate transformer.
        ds = roms.transform(z)
        """
        # TODO
        return

    # Depth related property decorators
    @property
    def z_r_rho(self):
        if '_z_r_rho' not in self._obj.coords:
            self._obj.coords['_z_r_rho'] = _calc_z(self._obj, '_r', '_rho')
        return self._obj.coords['_z_r_rho']

    @property
    def z_w_rho(self):
        if '_z_w_rho' not in self._obj.coords:
            self._obj.coords['_z_w_rho'] = _calc_z(self._obj, '_w', '_rho')
        return self._obj.coords['_z_w_rho']

    @property
    def z_r_u(self):
        if '_z_r_u' not in self._obj.coords:
            self._obj.coords['_z_r_u'] = _calc_z(self._obj, '_r', '_u')
        return self._obj.coords['_z_r_u']

    @property
    def z_w_u(self):
        if '_z_w_u' not in self._obj.coords:
            self._obj.coords['_z_w_u'] = _calc_z(self._obj, '_w', '_u')
        return self._obj.coords['_z_w_u']

    @property
    def z_r_v(self):
        if '_z_r_v' not in self._obj.coords:
            self._obj.coords['_z_r_v'] = _calc_z(self._obj, '_r', '_v')
        return self._obj.coords['_z_r_v']

    @property
    def z_w_v(self):
        if '_z_w_v' not in self._obj.coords:
            self._obj.coords['_z_w_v'] = _calc_z(self._obj, '_w', '_v')
        return self._obj.coords['_z_w_v']

    @property
    def z_r(self):
        pos = self._hpos[0]
        var = '_z_r' + pos
        if var not in self._obj.coords:
            self._obj.coords[var] = _calc_z(self._obj, '_r', pos)
        return self._obj.coords[var]

    @property
    def z_w(self):
        pos = self._hpos[0]
        var = '_z_w' + pos
        if var not in self._obj.coords:
            self._obj.coords[var] = _calc_z(self._obj, '_w', pos)
        return self._obj.coords[var]

    @property
    def z_rho(self):
        pos = self._vpos[0]
        var = '_z' + pos + '_rho'
        if var not in self._obj.coords:
            self._obj.coords[var] = _calc_z(self._obj, pos, '_rho')
        return self._obj.coords[var]

    @property
    def z_u(self):
        pos = self._vpos[0]
        var = '_z' + pos + '_u'
        if var not in self._obj.coords:
            self._obj.coords[var] = _calc_z(self._obj, pos, '_u')
        return self._obj.coords[var]

    @property
    def z_v(self):
        pos = self._vpos[0]
        var = '_z' + pos + '_v'
        if var not in self._obj.coords:
            self._obj.coords[var] = _calc_z(self._obj, pos, '_v')
        return self._obj.coords[var]

    @property
    def z(self):
        hpos, vpos = self._hpos[0], self._vpos[0]
        var = '_z' + vpos + hpos
        if var not in self._obj.coords:
            self._obj.coords[var] = _calc_z(self._obj, vpos, hpos)
        return self._obj.coords[var]

    @property
    def dz_rho(self):
        if '_dz_rho' not in self._obj:
            _dz = self.z_w_rho.diff(dim='s_w')
            _dz = _dz.rename(s_w='s_rho').assign_coords(
                s_rho=self._obj.s_rho)
            self._obj['_dz_rho'] = _dz
        return self._obj['_dz_rho']

    @property
    def dz_u(self):
        if '_dz_u' not in self._obj:
            _dz = self.z_w_u.diff(dim='s_w')
            _dz = _dz.rename(s_w='s_rho').assign_coords(
                s_rho=self._obj.s_rho)
            self._obj['_dz_u'] = _dz
        return self._obj['_dz_u']

    @property
    def dz_v(self):
        if '_dz_v' not in self._obj:
            _dz = self.z_w_v.diff(dim='s_w')
            _dz = _dz.rename(s_w='s_rho').assign_coords(
                s_rho=self._obj.s_rho)
            self._obj['_dz_v'] = _dz
        return self._obj['_dz_v']

    @property
    def dz(self):
        hpos = self._hpos[0]
        if hpos == '_rho':
            return self.dz_rho
        elif hpos == '_u':
            return self.dz_u
        elif hpos == '_v':
            return self.dz_v
        else:
            if '_dz' not in self._obj:
                _dz = self.z_w.diff(dim='s_w')
                _dz = _dz.rename(s_w='s_rho').assign_coords(
                    s_rho=self._obj.s_rho)
                self._obj['_dz'] = _dz
            return self._obj['_dz']

    @property
    def vol(self):
        if '_vol' not in self._obj:
            self._obj['_vol'] = self.dz_rho/(self._obj.pm*self._obj.pn)
        return self._obj['_vol']

    # Other commonly used alias
    xi = property(lambda self: self._obj['xi' + self._hpos[0]])
    eta = property(lambda self: self._obj['eta' + self._hpos[0]])
    x = property(lambda self: self._obj['x' + self._hpos[0]])
    y = property(lambda self: self._obj['y' + self._hpos[0]])
    lon = property(lambda self: self._obj['lon' + self._hpos[0]])
    lat = property(lambda self: self._obj['lat' + self._hpos[0]])

    # Linker of the plotting function
    plot = property(lambda self: _ROMSDatasetPlot(self._obj))


class _ROMSDatasetPlot:
    """
    This class wraps DataArray.plot
    """

    def __init__(self, obj):
        self._obj = obj

    def _plot_decorator(self, func):
        """
        This decorator is used to set default kwargs on plotting functions.
        For now it put s_rho on Y axis when data is mapped.
        (1) put s_rho/w on Y axis when data is mapped on X-Z or Y-Z direction.
        (2) set 'distance' on x axis if data is plotted along transect.
        (3) set colormap to RdBu_r if vmin = -vmax
        (4) Plot with geometric coordinates instead of numeric coordinates.

        By passing in keyword argument geo=True/False, the plot method will
        use lat-lon/x-y coordinates in mapped plots.
        """

        @functools.wraps(func)
        def _plot_wrapper(*args, **kwargs):

            z_coord_names = ['depth', 'z', 'vertical']

            # args is passed in as tuple. Need to convert to list to
            # allow insertion/value change.
            largs = [i for i in args]

            if func.__name__ in ['quiver', ]:
                # For now, use the coordinate of the 'x' direction value as
                # the plot reference coordinate.
                hpos = _find_hpos(self._obj[args[-2]])
                vpos = _find_vpos(self._obj[args[-2]])
                assert hpos == _find_hpos(self._obj[args[-1]]), \
                    'U/V must be on the same coord system.'
                dims = list(self._obj[args[-2]].dims)
                coords = list(self._obj[args[-2]].coords)

                # Find which vertical coordinates to convert
                zcor = '_z' + vpos + hpos
                if 's_rho' in dims:
                    scor = 's_rho'
                elif 's_w' in dims:
                    scor = 's_w'
                else:
                    scor = None

                # quiver (and streamline) plot takes exactly 4 input args.
                # If less than 4 arguments are passed in, need to expand
                # args by inserting default values.
                if len(args) == 2:
                    # If 2 input arguments, check the dims of DataArrays
                    # for vertical coordinates s_rho/s_w. If s_rho/s_w is
                    # in dims, insert the first argument as horizontal coord
                    # and second as vertical coord; otherwise pass the two
                    # dims of DataArray directly to args.
                    largs.insert(0, dims[0])
                    largs.insert(0, dims[1])
                    if 'track' in dims:
                        largs[0] = 'distance'
                        dims.remove('track')
                        if dims[0] in ['s_rho', 's_w']:
                            largs[1] = zcor
                        else:
                            largs[1] = dims[0]
                    elif scor in dims:
                        largs[1] = zcor
                        dims.remove(scor)
                        largs[0] = dims[0]

                elif len(args) == 3:
                    # If 3 input arguments, check the first argument [x],
                    # and insert z_r/z_w as the second argument if it is
                    # not used by x axis. If [x] uses z_r/z_w/s_rho, set
                    # the second argument to the unused dimension.
                    if args[0] in dims:
                        dims.remove(args[0])
                        if dims[0] == scor:
                            largs.insert(1, zcor)
                        elif dims[0] == 'track':
                            largs.insert(0, 'distance')
                        else:
                            largs.insert(0, dims[0])
                    elif args[0] not in dims and args[0] in z_coord_names:
                        if scor in dims:
                            largs[0] = zcor
                            dims.remove(scor)
                            if dims[0] == 'track':
                                largs.insert(0, 'distance')
                            else:
                                largs.insert(0, dims[0])
                        else:
                            raise ValueError()
                    elif args[0] == 'distance':
                        dims.remove('track')
                        if dims[0] == scor:
                            largs.insert(1, zcor)
                        else:
                            largs.insert(0, dims[0])
                    else:
                        raise ValueError('Invalid coordiante argument ' +
                                         args[0])

                elif len(args) == 4:
                    # If 4 input arguments, check if the first/second
                    # argument [x/y] is s_rho/s_w and replace it with
                    # z_r/z_w if use vertical coords.
                    if args[1] not in coords and \
                       args[1] in z_coord_names:
                        largs[1] = zcor
                    elif args[0] not in coords and \
                            args[0] in z_coord_names:
                        largs[0] = zcor

            # set colormap to RdBu_r
            if 'vmin' in kwargs and 'vmax' in kwargs and \
               kwargs['vmin'] == -kwargs['vmax']:
                kwargs['cmap'] = 'RdBu_r'

            # Extract keyword arguments used by the wrapper
            # If geo=True, use lat/lon coords; if geo=False, use x/y
            # coords; if geo not provided, use xi/eta coords.
            if 'geo' in kwargs:
                geo = kwargs.pop('geo')
                for i, l in enumerate(largs):
                    if geo:
                        xcor, ycor = 'lon', 'lat'
                    else:
                        xcor, ycor = 'x', 'y'
                    if 'xi' in l:
                        largs[i] = xcor + l.split('xi')[-1]
                    if 'eta' in l:
                        largs[i] = ycor + l.split('eta')[-1]

            # TODO Fix streamline plots. Since pyplot.streamplot only accepts
            # rect-linear grids, extra step is needed to convert from s-coord
            # to z-coord rect grid.
            # if func.__name__ == 'streamplot':
            #     # If streamplot and _z_r or _z_w used in one dimension,
            #     # need to replicate the other dimension since streamplot
            #     # takes 2-D mesh if one of the coordinate parameter is 2-D.
            #     xcor = largs[xdim]
            #     xcor2d = '_' + xcor + '2d'
            #     if largs[zdim] == '_z_r':
            #         self._obj.coords[xcor2d] = \
            #             self._obj[xcor] * xr.ones_like(self._obj.s_rho)
            #     elif largs[zdim] == '_z_w':
            #         self._obj.coords[xcor2d] = \
            #             self._obj[xcor] * xr.ones_like(self._obj.s_w)
            #     largs[xdim] = xcor2d

            # Compute depth coordinates z_r/z_w if used by the plot method
            if zcor not in self._obj.coords:
                self._obj.coords[zcor] = _calc_z(self._obj, vpos, hpos)

            return func(*largs, **kwargs)

        return _plot_wrapper

    def __call__(self, *args, **kwargs):
        """
        Fallback to Dataset.plot()
        """
        return self._obj.plot()

    def __getattr__(self, attr):
        """
        Wraps xarray.plot.** methods.
        """
        func = getattr(self._obj.plot, attr)
        return self._plot_decorator(func)


@xr.register_dataarray_accessor("roms")
class ROMSDataArrayAccessor(ROMSAccessor):

    def __init__(self, obj):
        """
        Pass in a DataArray object for ROMSDataArrayAccessor.
        """
        super().__init__(obj)
        self._hpos = _find_hpos(obj)
        self._vpos = _find_vpos(obj)

    def interp(self, lon, lat, time=None):
        """
        Horizontal interpolation method for ROMS variable DataArray.

        da = roms.interp(lon, lat, time=None)
        Inputs:
            lon, lat - 1-D list of longitude/latitude
            time (optional) - 1-D list of time
        Output:
            da - interpolated DataArray.
        """
        ds_locs = self.set_locs(lon, lat, time)

        # Calculate disctance array and find the integer part of coordiantes.
        xdis = self.x - ds_locs['x']
        ydis = self.y - ds_locs['y']
        dis = np.hypot(xdis, ydis).argmin(dim=(self.eta_nam, self.xi_nam))

        etav, xiv = dis[self.eta_nam], dis[self.xi_nam]
        eta0, xi0 = self.eta.data[0], self.xi.data[0]

        etaN, xiN = self.eta.size, self.xi.size
        etav = etav.where(etav < etaN-2, etaN-2)
        xiv = xiv.where(xiv < xiN-2, xiN-2)

        eta_loc, xi_loc = [], []
        xv, yv = self.x.data, self.y.data

        for ei, xi, xx, yy in \
                zip(etav.data, xiv.data, ds_locs.x.data, ds_locs.y.data):
            ivec = np.array([xv[ei+1, xi]-xv[ei, xi], yv[ei+1, xi]-yv[ei, xi]])
            jvec = np.array([xv[ei, xi+1]-xv[ei, xi], yv[ei, xi+1]-yv[ei, xi]])
            c = np.array([xx-xv[ei, xi], yy-yv[ei, xi]])
            efrac = np.dot(ivec, c)/(np.dot(ivec, ivec))
            xfrac = np.dot(jvec, c)/(np.dot(jvec, jvec))

            eta_loc.append(ei + eta0 + efrac)
            xi_loc.append(xi + xi0 + xfrac)

        ds_locs['eta'] = DataArray(eta_loc, dims=('track'))
        ds_locs['xi'] = DataArray(xi_loc, dims=('track'))
        interp_coords = {self.eta_nam: ds_locs.eta,
                         self.xi_nam: ds_locs.xi}
        if time is not None:
            interp_coords['ocean_time'] = ds_locs.ocean_time
        da = self._obj.interp(interp_coords)

        # Clean up coordiantes
        drop_coords = []
        for coord in da.coords:
            if coord != 's_rho' and self._hpos in coord:
                drop_coords.append(coord)
        for coord in drop_coords:
            new_name = coord.split(self._hpos)[0]
            if new_name not in da.coords:
                da = da.rename({coord: new_name})
            else:
                da.__delitem__(coord)

        return da

    def transform(self, z):
        """
        Vertical coordinate transformer.

        da = roms.transform(z)
        """
        z = DataArray(-np.abs(z), dims='Z')
        grd = Grid(self._obj, coords={'S': {'center': self.s}},
                   periodic=False)
        da = grd.transform(self._obj, 'S', z, target_data=self.z)
        return da

    def loc_selector(self):
        """
        GUI interpolation location selector.
        """
        self.loc = {}
        lon, lat = [], []
        land_color = (0.3, 0.3, 0.3)
        sea_color = (0.0, 0.0, 0.0, 0)
        # ice_color = (0.9, 0.9, 0.9, 0.5)
        fig, ax = plt.subplots()
        self.h.roms.plot.pcolormesh(geo=True, ax=ax, vmin=0, cmap='YlGn')
        cmap = plt.matplotlib.colors.ListedColormap(
            [land_color, sea_color], name='land/sea')
        self.mask.roms.plot.pcolormesh(
            geo=True, ax=ax, vmin=0, vmax=1, cmap=cmap, add_colorbar=False)
        fig.tight_layout()
        plt.show()
        self.loc['lon'], self.loc['lat'] = lon, lat
        return

    def interp_gui(self):
        """
        GUI interpolation track selector.
        """
        # TODO Finish this.
        lon, lat = [], []
        land_color = (0.3, 0.3, 0.3)
        sea_color = (0.0, 0.0, 0.0, 0)
        # ice_color = (0.9, 0.9, 0.9, 0.5)
        fig, ax = plt.subplots()
        self.h.roms.plot.pcolormesh(geo=True, ax=ax, vmin=0, cmap='YlGn')
        cmap = plt.matplotlib.colors.ListedColormap(
            [land_color, sea_color], name='land/sea')
        self.mask.roms.plot.pcolormesh(
            geo=True, ax=ax, vmin=0, vmax=1, cmap=cmap, add_colorbar=False)
        fig.tight_layout()
        plt.show()
        da = self.interp(lon, lat)
        return da

    # Depth-related decorator properties
    @property
    def z(self):
        var = '_z' + self._vpos + self._hpos
        if var not in self._obj.coords:
            self._obj.coords[var] = _calc_z(self._obj, self._vpos, self._hpos)
        return self._obj[var]

    @property
    def s_r(self):
        if self._vpos == '_r':
            return self._obj['s_rho']
        else:
            return None

    # Other commonly used property decorators
    hpos = property(lambda self: self._hpos)
    vpos = property(lambda self: self._vpos)

    @property
    def s_nam(self):
        if self._vpos == '_r':
            return 's_rho'
        else:
            return 's' + self._vpos
    xi_nam = property(lambda self: 'xi' + self._hpos)
    eta_nam = property(lambda self: 'eta' + self._hpos)

    s = property(lambda self: self._obj[self.s_nam])
    xi = property(lambda self: self._obj[self.xi_nam])
    eta = property(lambda self: self._obj[self.eta_nam])
    x = property(lambda self: self._obj['x' + self._hpos])
    y = property(lambda self: self._obj['y' + self._hpos])
    lon = property(lambda self: self._obj['lon' + self._hpos])
    lat = property(lambda self: self._obj['lat' + self._hpos])
    h = property(lambda self: self._obj['h' + self._hpos])
    mask = property(lambda self: self._obj['mask' + self._hpos])

    # Linker of the plotting function
    plot = property(lambda self: _ROMSDataArrayPlot(self._obj,
                                                    self._vpos, self._hpos))


class _ROMSDataArrayPlot:
    """
    This class wraps DataArray.plot
    """

    def __init__(self, obj, vpos, hpos):
        self._obj = obj
        self._vpos = vpos
        self._hpos = hpos

    def _plot_decorator(self, func):
        """
        This decorator is used to set default kwargs on plotting functions.
        For now, it
        (1) put s_rho/w on Y axis when data is mapped on X-Z or Y-Z direction.
        (2) set 'distance' on x axis if data is plotted along transect.
        (3) set colormap to RdBu_r if vmin = -vmax
        (4) Plot with geometric coordinates instead of numeric coordinates.

        By passing in keyword argument geo=True/False, the plot method will
        use lat-lon/x-y coordinates in mapped plots.
        """

        @functools.wraps(func)
        def _plot_wrapper(*args, **kwargs):

            z_coord_names = ['depth', 'z', 'vertical']

            # Find default depth coordinate references
            dims = list(self._obj.dims)

            # Find which vertical coordinates to convert
            if self._vpos == '_r':
                scor, zcor = 's_rho', '_z' + self._vpos + self._hpos
            elif self._vpos == '_w':
                scor, zcor = 's_w', '_z' + self._vpos + self._hpos
            else:
                scor, zcor = None, None

            if func.__name__ in ['contour', 'contourf', 'pcolormesh',
                                 'surface']:

                # Set default keyword arguments
                if 'x' not in kwargs and 'y' not in kwargs:
                    if scor in dims:
                        kwargs['y'] = zcor
                        dims.remove(scor)
                        if 'track' == dims[0]:
                            kwargs['x'] = 'distance'
                        else:
                            kwargs['x'] = dims[0]
                    else:
                        if 'track' in dims:
                            kwargs['x'] = 'distance'
                            dims.remove('track')
                            kwargs['y'] = dims[0]
                        else:
                            kwargs['x'], kwargs['y'] = dims[1], dims[0]
                elif 'x' not in kwargs and 'y' in kwargs:
                    if kwargs['y'] in z_coord_names:
                        kwargs['y'] = zcor
                        dims.remove(scor)
                        if dims[0] == 'track':
                            kwargs['x'] = 'distance'
                        else:
                            kwargs['x'] = dims[0]
                    else:
                        if dims[1] == scor:
                            kwargs['x'] = zcor
                        elif dims[1] == 'track':
                            kwargs['x'] = 'distance'
                elif 'x' in kwargs and 'y' not in kwargs:
                    if kwargs['x'] in z_coord_names:
                        kwargs['x'] = zcor
                        dims.remove(scor)
                        if dims[0] == 'track':
                            kwargs['y'] = 'distance'
                        else:
                            kwargs['y'] = dims[0]
                    else:
                        if dims[0] == scor:
                            kwargs['y'] = zcor
                        elif dims[0] == 'track':
                            kwargs['y'] = 'distance'
                else:
                    if kwargs['x'] in z_coord_names:
                        kwargs['x'] = zcor
                    if kwargs['y'] in z_coord_names:
                        kwargs['y'] = zcor

            elif func.__name__ in ['line']:
                if 'hue' in kwargs and kwargs['hue'] in z_coord_names:
                    kwargs['hue'] = zcor
                elif 'x' in kwargs and kwargs['x'] in z_coord_names:
                    kwargs['x'] = zcor
                elif 'y' in kwargs and kwargs['y'] in z_coord_names:
                    kwargs['y'] = zcor

                if 'hue' in kwargs and kwargs['hue'] == 'track':
                    kwargs['hue'] = 'distance'
                elif 'x' in kwargs and kwargs['x'] == 'track':
                    kwargs['x'] = 'distance'
                elif 'y' in kwargs and kwargs['y'] == 'track':
                    kwargs['y'] = 'distance'

            for k in kwargs:
                if kwargs[k] == zcor:
                    self._obj.roms.z

            # set colormap to RdBu_r
            if 'vmin' in kwargs and 'vmax' in kwargs:
                if kwargs['vmin'] == -kwargs['vmax']:
                    kwargs['cmap'] = 'RdBu_r'

            # Extract keywords used by the wrapper.
            # If geo=True, use lat/lon coords; if geo=False, use x/y
            # coords; if geo not provided, use xi/eta coords.
            if 'geo' in kwargs:
                geo = kwargs.pop('geo')
                for k in kwargs.keys():
                    if geo:
                        xcor, ycor = 'lon', 'lat'
                    else:
                        xcor, ycor = 'x', 'y'
                    if kwargs[k] in ['xi_rho', 'xi_psi', 'xi_u', 'xi_v']:
                        kwargs[k] = xcor + kwargs[k].split('xi')[-1]
                    elif kwargs[k] in ['eta_rho', 'eta_psi', 'eta_u', 'eta_v']:
                        kwargs[k] = ycor + kwargs[k].split('eta')[-1]

            return func(*args, **kwargs)

        return _plot_wrapper

    def __call__(self, *args, **kwargs):
        """
        Allows direct plot with DataArray.plot()
        """

        # Set default plot method
        if len(self._obj.dims) == 1:
            plot = self._obj.plot.line
        elif len(self._obj.dims) == 2:
            plot = self._obj.plot.pcolormesh
        else:
            plot = self._obj.plot.hist
        return self._plot_decorator(plot)(*args, **kwargs)

    def __getattr__(self, attr):
        """
        Wraps xarray.plot.** methods.
        """
        func = getattr(self._obj.plot, attr)
        return self._plot_decorator(func)


def open_dataset(filename, grid_filename=None, interp_rho=False, **kwargs):
    """
    Open a ROMS history/average file, optionally a ROMS grid file, and
    construct a RDataset.

    ds = open_dataset(filename, grid_filename=None, interp_rho=False,
                      **kwargs)
    Inputs:
        filename      - str, ROMS file path
        grid_filename - str, ROMS grid file path
        interp_rho    - bool, if to interpolate all dataArray to rho points
        **kwargs      - other keyword arguments to be passed to RDataset
    """
    ds = xr.open_dataset(filename, **kwargs)
    if grid_filename is not None:
        grd = grid.get_ROMS_grid(grid_file=grid_filename,
                                 hist_file=filename)
        if 'zeta' in ds:
            grd.vgrid.zeta = ds.zeta
        return RDataset(ds, grd, interp_rho=interp_rho)
    else:
        return RDataset(ds, interp_rho=interp_rho)


class RDataset(xr.Dataset):
    """
    ROMS Xarray Dataset object.
    This is a subclass of Xarray Dataset. This class wraps roms history/average
    files and sets coordinates, either using a pyROMS grid object or coordinate
    info from the history/average file itself.

    Optionally it interpolates data at u/v points to rho points.

    Usage:
        dsr = RDataset(grid, *args, **kwargs)

        grid             - pyROMS grid object
        *args, ** kwargs - other input arguments to pass to xr.Dataset

    Or:
        dsr = RDataset(ds[, grid], interp_rho=False)

        ds         - Dataset of roms file(s) loaded with xr.open_(mf)dataset()
        grid       - pyROMS grid object
        interp_rho - switch to enable interpolation to rho-points
    """

    __slots__ = ('_grid', '_interp_rho')

    def __init__(self, *args, **kwargs):
        """
        Initialize a RDataset object.
        """

        # Process input arguments.
        # If pyroms grid object is provided in input arguments, fetch it.
        use_grid = False
        if 'grid' in kwargs.keys():
            self._grid = kwargs.pop('grid')
            use_grid = True
        else:
            for i in args:
                if isinstance(i, grid.ROMSGrid):
                    self._grid = i
                    use_grid = True

        # If interp_rho in input args, pass it in. Otherwise set it to False.
        if 'interp_rho' in kwargs.keys():
            self._interp_rho = kwargs.pop('interp_rho')
        else:
            self._interp_rho = False

        # If a Xarray Dataset object is passed in, initialize with Dataset
        # and ignore all other input arguments. Otherwise initialize with
        # input arguments.
        if isinstance(args[0], xr.Dataset):
            super().__init__(args[0].data_vars, args[0].coords, args[0].attrs)
        else:
            super().__init__(*args, **kwargs)

        # ---------------------------------------------------------------
        # Check and write grid information to dataset.
        if use_grid:
            self.coords['_spherical'] = self._grid.hgrid.spherical
            dsg = self._grid.to_xarray()
            # Merge dsg into dsr. Since some variables in dsg is determined
            # as coordinates, cannot use the internal xr.merge method.
            for var in dsg.variables:
                if var not in self and var not in self.coords:
                    self.coords[var] = dsg[var]
        else:
            # No grid file provided. Need to extract coords from ROMS file
            # and calcuate x/y coords.
            print('No grid file provided. Will generate a temporary grid.')
            # Check if ds contains grid information, which is required for
            # computing z_r/z_w.
            assert hasattr(self, 'h'), 'No grid information provided in ' + \
                                       'ROMS file. Need to provide grid file.'

            if 'x_rho' in self:
                self.coords['_spherical'] = False
                # Add coords to exsisting CGrid.
                x, y = hgrid.rho_to_vert(
                    self.x_rho, self.y_rho, self.pm, self.pn, self.angle)
                self.coords['x_vert'] = (['eta_vert', 'xi_vert'], x)
                self.coords['y_vert'] = (['eta_vert', 'xi_vert'], y)
                hgrd = hgrid.CGrid(x, y)
            else:
                self.coords['_spherical'] = True
                # Construct a temporary Geo CGrid.
                eta0 = int(self.dims['eta_rho']/2)
                xi0 = int(self.dims['xi_rho']/2)
                lon0 = self.lon_rho[eta0, xi0].item()
                lat0 = self.lat_rho[eta0, xi0].item()
                proj = pyproj.Proj(proj='aeqd', lat_0=lat0, lon_0=lon0)

                x, y = proj(self.lon_rho, self.lat_rho)
                self.coords['x_rho'] = (['eta_rho', 'xi_rho'], x)
                self.coords['y_rho'] = (['eta_rho', 'xi_rho'], y)

                x, y = proj(self.lon_psi, self.lat_psi)
                self.coords['x_psi'] = (['eta_psi', 'xi_psi'], x)
                self.coords['y_psi'] = (['eta_psi', 'xi_psi'], y)
                x, y = proj(self.lon_u, self.lat_u)
                self.coords['x_u'] = (['eta_u', 'xi_u'], x)
                self.coords['y_u'] = (['eta_u', 'xi_u'], y)
                x, y = proj(self.lon_v, self.lat_v)
                self.coords['x_v'] = (['eta_v', 'xi_v'], x)
                self.coords['y_v'] = (['eta_v', 'xi_v'], y)

                # vert coords
                lon, lat = hgrid.rho_to_vert_geo(
                    self.lon_rho.data, self.lat_rho.data,
                    self.lon_psi.data, self.lat_psi.data,
                    proj=proj)
                x, y = proj(lon, lat)
                self.coords['lon_vert'] = (['eta_vert', 'xi_vert'], lon)
                self.coords['lat_vert'] = (['eta_vert', 'xi_vert'], lat)
                self.coords['x_vert'] = (['eta_vert', 'xi_vert'], x)
                self.coords['y_vert'] = (['eta_vert', 'xi_vert'], y)

                hgrd = hgrid.CGridGeo(lon, lat, proj)

            vgrd = vgrid.SCoord(
                self.h.data, self.theta_b.data,
                self.theta_s.data, self.Tcline.data,
                self.dims['s_rho'],
                self.Vtransform.item(), self.Vstretching.item())
            if 'zeta' in self:
                vgrd.zeta = self.zeta
            if 'zice' in self:
                vgrd.zice = self.zice

            self._grid = grid.ROMSGrid(
                'Generated by pyroms.xr', hgrd, vgrd)

        # ---------------------------------------------------------------
        # Remake basic coordinates for easy interpolation
        self.coords['eta_rho'] = np.arange(self.dims['eta_rho'])+0.5
        self.coords['xi_rho'] = np.arange(self.dims['xi_rho'])+0.5
        self.coords['eta_vert'] = range(self.dims['eta_rho']+1)
        self.coords['xi_vert'] = range(self.dims['xi_rho']+1)
        self.coords['eta_psi'] = range(self.dims['eta_psi'])
        self.coords['xi_psi'] = range(self.dims['xi_psi'])
        self.coords['eta_u'] = self.eta_rho.data.copy()
        self.coords['xi_u'] = self.xi_psi.data.copy()
        self.coords['eta_v'] = self.eta_psi.data.copy()
        self.coords['xi_v'] = self.xi_rho.data.copy()

        # ---------------------------------------------------------------
        # Perform interpolation to RHO points or generate coords for u/v
        # points
        if self._interp_rho:
            # Interpolate and remove u/v related coords
            itp = self.interp(eta_u=self.eta_rho, eta_v=self.eta_rho,
                              xi_u=self.xi_rho, xi_v=self.xi_rho,
                              s_w=self.s_rho)
            for var in self.data_vars:
                self[var] = itp[var]
            drop_vars = []
            for var in self:
                if 'ocean_time' not in self[var].dims and \
                   ('_u' in var or '_v' in var or '_w' in var):
                    drop_vars.append(var)
            for var in drop_vars:
                self.__delitem__(var)
        else:
            # Interpolate bathymetry variables to u/v points
            self['h_u'] = self.h.interp(
                eta_rho=self.eta_u, xi_rho=self.xi_u)
            self['h_v'] = self.h.interp(
                eta_rho=self.eta_v, xi_rho=self.xi_v)
            if 'zeta' in self:
                self['zeta_u'] = self.zeta.interp(
                    eta_rho=self.eta_u, xi_rho=self.xi_u)
                self['zeta_v'] = self.zeta.interp(
                    eta_rho=self.eta_v, xi_rho=self.xi_v)
            else:
                self['zeta'] = xr.zeros_like(self.h)
                self['zeta_u'] = xr.zeros_like(self.h_u)
                self['zeta_v'] = xr.zeros_like(self.h_v)
            if 'zice' in self:
                self['zice_u'] = self.zice.interp(
                    eta_rho=self.eta_u, xi_rho=self.xi_u)
                self['zice_v'] = self.zice.interp(
                    eta_rho=self.eta_v, xi_rho=self.xi_v)
            else:
                self['zice'] = xr.zeros_like(self.h)
                self['zice_u'] = xr.zeros_like(self.h_u)
                self['zice_v'] = xr.zeros_like(self.h_v)

            if 'mask_rho' in self:
                self['mask'] = self.mask_rho
            if 'mask_is' in self:
                self['mask_is_u'] = self.mask_is.interp(
                    eta_rho=self.eta_u, xi_rho=self.xi_u)
                self['mask_is_v'] = self.mask_is.interp(
                    eta_rho=self.eta_v, xi_rho=self.xi_v)
                self['mask_is_u'] = self.mask_is_u.where(
                    self.mask_is_u == 0, 1)
                self['mask_is_v'] = self.mask_is_v.where(
                    self.mask_is_v == 0, 1)
            else:
                self['mask_is'] = xr.zeros_like(self.h)
                self['mask_is_u'] = xr.zeros_like(self.h_u)
                self['mask_is_v'] = xr.zeros_like(self.h_v)

        # ---------------------------------------------------------------
        # House cleaning.
        # Pass in projeciton information to Dataset, in order to generate
        # projection function for later use.
        for var in self.data_vars:
            proj4_init = self._grid.hgrid.proj.to_proj4()
        self.attrs['proj4_init'] = proj4_init

        # When Dataset is loaded with open_mfdataset, clean some variables
        # by averaging over ocean_time.
        if self.ntimes.dims == ('ocean_time', ):
            for var in self.coords:
                vdim = self[var].dims
                if var != 'ocean_time' and 'ocean_time' in vdim:
                    if vdim == ('ocean_time', ) or \
                       vdim == ('ocean_time', 'tracer', ) or \
                       vdim == ('ocean_time', 'boundary', ) or \
                       vdim == ('ocean_time', 'boundary', 'tracer', ) or \
                       vdim == ('ocean_time', 's_rho', ) or \
                       vdim == ('ocean_time', 's_w', ) or \
                       var in ['h', 'h_rho', 'h_u', 'h_v',
                               'h_psi', 'h_vert',
                               'zice', 'zice_rho', 'zice_u', 'zice_v',
                               'zice_psi', 'zice_vert',
                               'mask', 'mask_rho', 'mask_u', 'mask_v',
                               'mask_psi', 'mask_vert',
                               'f', 'angle']:
                        self.coords[var] = self[var].isel(ocean_time=0)

            for var in self:
                vdim = self[var].dims
                if var != 'ocean_time' and 'ocean_time' in vdim:
                    if vdim == ('ocean_time', ) or \
                       vdim == ('ocean_time', 'tracer', ) or \
                       vdim == ('ocean_time', 'boundary', ) or \
                       vdim == ('ocean_time', 'boundary', 'tracer', ) or \
                       vdim == ('ocean_time', 's_rho', ) or \
                       vdim == ('ocean_time', 's_w', ) or \
                       var in ['h', 'h_rho', 'h_u', 'h_v',
                               'h_psi', 'h_vert',
                               'zice', 'zice_rho', 'zice_u', 'zice_v',
                               'zice_psi', 'zice_vert',
                               'mask', 'mask_rho', 'mask_u', 'mask_v',
                               'mask_psi', 'mask_vert',
                               'f', 'angle']:
                        self[var] = self[var].isel(ocean_time=0)

        # Transform some data variables to coordinates and some other
        # model constants to attributes.
        drop_vars = []
        for var in self.data_vars:
            vdim = self[var].dims
            if 'ocean_time' not in self[var].dims:
                # if (len(vdim) > 0 and
                #     vdim != ('tracer', ) and 'boundary' not in vdim) or \
                if (len(vdim) > 0) or \
                   var in ['Vtransform', 'Vstretching',
                           'theta_s', 'theta_b', 'Tcline', 'hc']:
                    self.coords[var] = self[var]
                else:
                    drop_vars.append(var)
                    if len(vdim) == 0:
                        self.attrs[var] = self[var].item()
                    else:
                        self.attrs[var] = self[var].data
        for var in drop_vars:
            self.__delitem__(var)

        # If zeta is in data variables, move it to coordinates
        if 'zeta' in self.data_vars:
            self.coords['zeta'] = self.zeta
            if not self._interp_rho:
                self.coords['zeta_u'] = self.zeta_u
                self.coords['zeta_v'] = self.zeta_v

        # Alias of h, zeta, zice, for easy access by depth computations.
        self.coords['h_rho'] = self.h
        if 'zeta' in self.coords:
            self.coords['zeta_rho'] = self.zeta
        if 'zice' in self.coords:
            self.coords['zice_rho'] = self.zice

        if self._spherical:
            # Flip longitude if longitude has a very wide range, which suggests
            # the grid crosses 0 or 180 degree. If the grid is global, the
            # flipping is performed twice and becomes the original grid.
            if self.lon_rho.max() - self.lon_rho.min() > 355:
                self.roms.longitude_wrap()
            if self.lon_rho.max() - self.lon_rho.min() > 355:
                self.roms.longitude_wrap()
