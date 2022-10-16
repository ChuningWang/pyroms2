import functools
import numpy as np
import pyproj
import xarray as xr
from xarray import DataArray, Dataset
from numba import guvectorize
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
        s_nam = 's_rho'
    elif vpos == '_w':
        s_nam = 's_w'
    h, zice, zeta = _set_hzz(obj, hpos)
    z = _set_z(h, zice, zeta,
               obj[s_nam], obj['Cs' + vpos],
               obj.hc, obj.Vtransform)
    z.attrs = dict(
        long_name='z' + vpos + ' at ' + hpos[1:].upper() + '-points',
        units='meter')
    return z


@guvectorize(
    "(float64[:], float64[:], float64[:], float64[:])",
    "(n), (n), (m) -> (m)",
    nopython=True)
def _interp1d(data, zr, zw, out):
    """
    A simple 1-D interpolator.
    """
    out[:] = np.interp(zw, zr, data)
    return


class ROMSAccessor:
    """
    This is the basic ROMS accessor for both Xarray Dataset and DataArray.
    """

    def __init__(self, obj):
        self._obj = obj
        if isinstance(obj, Dataset):
            self._is_dataset = True
            self._is_dataarray = False
        elif isinstance(obj, DataArray):
            self._is_dataset = False
            self._is_dataarray = True
        else:
            self._is_dataset = False
            self._is_dataarray = False

    def set_locs(self, lon, lat, time=None, get_vert=True, geo=True):
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

        lon, lat = np.asarray(lon).squeeze(), np.asarray(lat).squeeze()
        assert len(lon) == len(lat), 'lon/lat must have the same length.'
        assert lon.ndim == 1, 'lon/lat must be 1-D arrays.'
        npts = len(lon)
        proj = pyproj.Proj(self._obj.attrs['proj4_init'])
        if geo:
            x, y = proj(lon, lat)
        else:
            x, y = lon, lat
            lon, lat = proj(x, y, inverse=True)

        # Pass in geographic information to a DataArray
        ds_locs = Dataset()
        ds_locs['lon'] = DataArray(lon, dims=('track'))
        ds_locs['lat'] = DataArray(lat, dims=('track'))
        ds_locs['x'] = DataArray(x, dims=('track'))
        ds_locs['y'] = DataArray(y, dims=('track'))

        if time is not None:
            time = np.asarray(time)
            if len(time) == npts:
                ds_locs['ocean_time'] = DataArray(time, dims=('track'))
            else:
                ds_locs['ocean_time'] = DataArray(time, dims=('ocean_time'))

        # Assign coordiantes and calculate distance from lon[0]/lat[0]
        dis = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]).cumsum()
        dis = np.concatenate((np.array([0]), dis))
        ds_locs = ds_locs.assign_coords(track=np.arange(npts)+0.5)
        ds_locs = ds_locs.assign_coords(distance=('track', dis))

        if get_vert:
            xv, yv = np.zeros(npts+1), np.zeros(npts+1)
            xv[1:-1], yv[1:-1] = 0.5*(x[1:] + x[:-1]), 0.5*(y[1:] + y[:-1])
            xv[0], xv[-1] = xv[1] - (x[1] - x[0]), xv[-2] + (x[-1] - x[-2])
            yv[0], yv[-1] = yv[1] - (y[1] - y[0]), yv[-2] + (y[-1] - y[-2])
            lonv, latv = proj(xv, yv, inverse=True)
            disv = np.hypot(xv[1:] - xv[:-1], yv[1:] - yv[:-1]).cumsum()
            disv = np.concatenate((np.array([0]), disv))
            if lon.max() > 180:
                lonv = lonv % 360
            ds_locs['lon_vert'] = DataArray(lonv, dims=('track_vert'))
            ds_locs['lat_vert'] = DataArray(latv, dims=('track_vert'))
            ds_locs['x_vert'] = DataArray(xv, dims=('track_vert'))
            ds_locs['y_vert'] = DataArray(yv, dims=('track_vert'))
            ds_locs = ds_locs.assign_coords(track_vert=np.arange(npts+1))
            ds_locs = ds_locs.assign_coords(distance_vert=('track_vert', disv))

        return ds_locs

    def set_locs2d(self, lon, lat, time=None, get_vert=True, geo=True):
        """
        Put lon/lat and x/y coords in a DataArray given lon, lat as
        tuple/list/ndarray.

        ds_locs = roms.set_locs(lon, lat, time=None)
        Inputs:
            lon, lat - 2-D array of longitude/latitude
            time (optional) - 1-D list of time
        Output:
            ds_locs - interpolation coordinates.
        """

        lon, lat = np.asarray(lon).squeeze(), np.asarray(lat).squeeze()
        if (lon.ndim == 1) & (lat.ndim == 1):
            lon, lat = np.meshgrid(lon, lat)
        assert lon.shape == lat.shape, 'lon/lat must have the same length.'
        assert lon.ndim == 2, 'lon/lat must be 2-D arrays.'
        nypts, nxpts = lon.shape
        proj = pyproj.Proj(self._obj.attrs['proj4_init'])
        if geo:
            x, y = proj(lon, lat)
        else:
            x, y = lon, lat
            lon, lat = proj(x, y, inverse=True)

        # Pass in geographic information to a DataArray
        ds_locs = Dataset()
        ds_locs['lon'] = DataArray(lon, dims=('yc', 'xc'))
        ds_locs['lat'] = DataArray(lat, dims=('yc', 'xc'))
        ds_locs['x'] = DataArray(x, dims=('yc', 'xc'))
        ds_locs['y'] = DataArray(y, dims=('yc', 'xc'))

        if time is not None:
            time = np.asarray(time)
            ds_locs['ocean_time'] = DataArray(time, dims=('ocean_time'))

        # Assign coordiantes
        ds_locs = ds_locs.assign_coords(yc=np.arange(nypts)+0.5)
        ds_locs = ds_locs.assign_coords(xc=np.arange(nxpts)+0.5)

        if get_vert:
            xx, yx = 0.5*(x[:, 1:]+x[:, :-1]), 0.5*(y[:, 1:]+y[:, :-1])
            xy, yy = 0.5*(x[1:, :]+x[:-1, :]), 0.5*(y[1:, :]+y[:-1, :])
            xv, yv = np.zeros((nypts+1, nxpts+1)), np.zeros((nypts+1, nxpts+1))
            xv[1:-1, 1:-1] = 0.25*(x[1:, 1:]+x[:-1, 1:]+x[1:, :-1]+x[:-1, :-1])
            yv[1:-1, 1:-1] = 0.25*(y[1:, 1:]+y[:-1, 1:]+y[1:, :-1]+y[:-1, :-1])
            xv[0, 1:-1] = 2*xx[0, :] - xv[1, 1:-1]
            yv[0, 1:-1] = 2*yx[0, :] - yv[1, 1:-1]
            xv[-1, 1:-1] = 2*xx[-1, :] - xv[-2, 1:-1]
            yv[-1, 1:-1] = 2*yx[-1, :] - yv[-2, 1:-1]
            xv[1:-1, 0] = 2*xy[:, 0] - xv[1:-1, 1]
            yv[1:-1, 0] = 2*yy[:, 0] - yv[1:-1, 1]
            xv[1:-1, -1] = 2*xy[:, -1] - xv[1:-1, -2]
            yv[1:-1, -1] = 2*yy[:, -1] - yv[1:-1, -2]
            xv[0, 0] = 4*x[0, 0] - xv[0, 1] - xv[1, 0] - xv[1, 1]
            yv[0, 0] = 4*y[0, 0] - yv[0, 1] - yv[1, 0] - yv[1, 1]
            xv[-1, 0] = 4*x[-1, 0] - xv[-1, 1] - xv[-2, 0] - xv[-2, 1]
            yv[-1, 0] = 4*y[-1, 0] - yv[-1, 1] - yv[-2, 0] - yv[-2, 1]
            xv[0, -1] = 4*x[0, -1] - xv[1, -1] - xv[0, -2] - xv[1, -2]
            yv[0, -1] = 4*y[0, -1] - yv[1, -1] - yv[0, -2] - yv[1, -2]
            xv[-1, -1] = 4*x[-1, -1] - xv[-1, -2] - xv[-2, -1] - xv[-2, -2]
            yv[-1, -1] = 4*y[-1, -1] - yv[-1, -2] - yv[-2, -1] - yv[-2, -2]
            lonv, latv = proj(xv, yv, inverse=True)
            if lon.max() > 180:
                lonv = lonv % 360
            angle_xy = np.arctan2(
                np.diff(0.5*(yv[1:, :]+yv[:-1, :])),
                np.diff(0.5*(xv[1:, :]+xv[:-1, :])))
            ds_locs['lon_vert'] = DataArray(lonv, dims=('yb', 'xb'))
            ds_locs['lat_vert'] = DataArray(latv, dims=('yb', 'xb'))
            ds_locs['x_vert'] = DataArray(xv, dims=('yb', 'xb'))
            ds_locs['y_vert'] = DataArray(yv, dims=('yb', 'xb'))
            ds_locs['angle_xyr'] = DataArray(angle_xy, dims=('yc', 'xc'))
            ds_locs = ds_locs.assign_coords(yb=np.arange(nypts+1))
            ds_locs = ds_locs.assign_coords(xb=np.arange(nxpts+1))

        return ds_locs

    def station(self, lon, lat, time=None):
        """
        Extract station profiles from ROMS Dataset.

        ds = roms.station(lon, lat)
        Inputs:
            lon, lat - float points of longitude/latitude
        Output:
            ds - station Dataset.
        """
        proj = pyproj.Proj(self._obj.attrs['proj4_init'])
        x, y = proj(lon, lat)

        # Calculate disctance array and find the integer part of coordiantes.
        if self._is_dataset:
            xdis, ydis = self._obj.x_rho - x, self._obj.y_rho - y
            dis = np.hypot(xdis, ydis).argmin(dim=('eta_rho', 'xi_rho'))
            eta0, xi0 = self._obj.eta_vert.data[0], self._obj.xi_vert.data[0]
            etav, xiv = dis['eta_rho'].item(), dis['xi_rho'].item()
            xv, yv = self._obj.x_vert.data, self._obj.y_vert.data
        elif self._is_dataarray:
            xdis, ydis = self.x - x, self.y - y
            dis = np.hypot(xdis, ydis).argmin(dim=(self.eta_nam, self.xi_nam))
            eta0, xi0 = self.eta.data[0], self.xi.data[0]
            etav, xiv = dis[self.eta_nam].item(), dis[self.xi_nam].item()
            etaN, xiN = self.eta.size-2, self.xi.size-2
            etav, xiv = min(etav, etaN), min(xiv, xiN)
            xv, yv = self.x.data, self.y.data

        # Calculate the fractional part of coordinates and add to integer part.
        ivec = np.array([xv[etav+1, xiv]-xv[etav, xiv],
                         yv[etav+1, xiv]-yv[etav, xiv]])
        jvec = np.array([xv[etav, xiv+1]-xv[etav, xiv],
                         yv[etav, xiv+1]-yv[etav, xiv]])
        c = np.array([x-xv[etav, xiv], y-yv[etav, xiv]])
        efrac = np.dot(ivec, c)/(np.dot(ivec, ivec))
        xfrac = np.dot(jvec, c)/(np.dot(jvec, jvec))
        eta = eta0 + etav + efrac
        xi = xi0 + xiv + xfrac

        # Perform interpolation using Xarray's interp method
        interp_coords = {}
        if self._is_dataset:
            interp_coords = {}
            for pos in self._hpos:
                interp_coords['eta' + pos] = eta
                interp_coords['xi' + pos] = xi
        elif self._is_dataarray:
            interp_coords = {self.eta_nam: eta,
                             self.xi_nam: xi}
        if time is not None:
            interp_coords['ocean_time'] = time
        dout = self._obj.interp(interp_coords)

        # Update and clean up coordiantes
        drop_coords = []
        if self._is_dataset:
            for var in ['lat', 'lon', 'x', 'y', 'mask']:
                dout.coords[var] = dout[var + '_rho']
            for coord in dout.coords:
                if ('_rho' in coord or '_u' in coord or '_v' in coord or
                    '_psi' in coord or '_vert' in coord) and \
                   coord != 's_rho':
                    drop_coords.append(coord)
            for coord in drop_coords:
                dout.__delitem__(coord)
        elif self._is_dataarray:
            for coord in dout.coords:
                if self._hpos in coord and coord != 's_rho':
                    drop_coords.append(coord)
            for coord in drop_coords:
                new_name = coord.split(self._hpos)[0]
                if new_name not in dout.coords:
                    dout = dout.rename({coord: new_name})
                else:
                    dout.__delitem__(coord)
        dout['eta'], dout['xi'] = eta, xi
        dout['lon'], dout['lat'], dout['x'], dout['y'] = lon, lat, x, y

        # Process angles to avoid wrap-around issue.
        angle = np.cos(self._obj.angle) + np.sin(self._obj.angle)*1j
        angle = angle.interp(dict(eta_rho=eta, xi_rho=xi))
        dout.angle.data = np.angle(angle)
        angle = np.cos(self._obj.angle_xy) + np.sin(self._obj.angle_xy)*1j
        angle = angle.interp(dict(eta_rho=eta, xi_rho=xi))
        dout.angle_xy.data = np.angle(angle)
        return dout

    def _indexer(self, ds_locs):
        # Calculate disctance array and find the integer part of coordiantes.
        # eta0, xi0 are the coords of left bottom corner.
        if self._is_dataset:
            xdis = self._obj.x_rho - ds_locs['x']
            ydis = self._obj.y_rho - ds_locs['y']
            dis = np.hypot(xdis, ydis).argmin(dim=('eta_rho', 'xi_rho'))
            eta0, xi0 = self._obj.eta_vert.data[0], self._obj.xi_vert.data[0]
            etav, xiv = dis['eta_rho'], dis['xi_rho']
            xv, yv = self._obj.x_vert.data, self._obj.y_vert.data
        elif self._is_dataarray:
            xdis, ydis = self.x - ds_locs['x'], self.y - ds_locs['y']
            dis = np.hypot(xdis, ydis).argmin(dim=(self.eta_nam, self.xi_nam))
            eta0, xi0 = self.eta.data[0], self.xi.data[0]
            etav, xiv = dis[self.eta_nam], dis[self.xi_nam]
            etaN, xiN = self.eta.size-2, self.xi.size-2
            etav = etav.where(etav < etaN, etaN)
            xiv = xiv.where(xiv < xiN, xiN)
            xv, yv = self.x.data, self.y.data

        # Calculate the fractional part of coordinates and add to integer part.
        eta_loc, xi_loc = [], []
        for ei, xi, xx, yy in \
                zip(etav.data, xiv.data, ds_locs.x.data, ds_locs.y.data):
            ivec = np.array([xv[ei+1, xi]-xv[ei, xi], yv[ei+1, xi]-yv[ei, xi]])
            jvec = np.array([xv[ei, xi+1]-xv[ei, xi], yv[ei, xi+1]-yv[ei, xi]])
            c = np.array([xx-xv[ei, xi], yy-yv[ei, xi]])
            efrac = np.dot(ivec, c)/(np.dot(ivec, ivec))
            xfrac = np.dot(jvec, c)/(np.dot(jvec, jvec))

            eta_loc.append(ei + eta0 + efrac)
            xi_loc.append(xi + xi0 + xfrac)

        eta = DataArray(eta_loc, dims=('track'))
        xi = DataArray(xi_loc, dims=('track'))
        return eta, xi

    def _interp(self, ds_locs):
        eta, xi = self._indexer(ds_locs)

        # Perform interpolation using Xarray's interp method
        if self._is_dataset:
            interp_coords = {}
            for pos in self._hpos:
                interp_coords['eta' + pos] = eta
                interp_coords['xi' + pos] = xi
        elif self._is_dataarray:
            interp_coords = {self.eta_nam: eta, self.xi_nam: xi}
        if 'ocean_time' in ds_locs:
            interp_coords['ocean_time'] = ds_locs.ocean_time
        dout = self._obj.interp(interp_coords)

        # Update and clean up coordiantes
        drop_coords = []
        if self._is_dataset:
            for var in ['lat', 'lon', 'x', 'y', 'mask']:
                dout.coords[var] = dout[var + '_rho']
            for coord in dout.coords:
                if ('_rho' in coord or '_u' in coord or '_v' in coord or
                    '_psi' in coord or '_vert' in coord) and \
                   coord != 's_rho':
                    drop_coords.append(coord)
            for coord in drop_coords:
                dout.__delitem__(coord)
        elif self._is_dataarray:
            for coord in dout.coords:
                if self._hpos in coord and coord != 's_rho':
                    drop_coords.append(coord)
            for coord in drop_coords:
                new_name = coord.split(self._hpos)[0]
                if new_name not in dout.coords:
                    dout = dout.rename({coord: new_name})
                else:
                    dout.__delitem__(coord)

        # Process angles to avoid wrap-around issue.
        angle = np.cos(self._obj.angle) + np.sin(self._obj.angle)*1j
        angle = angle.interp(dict(eta_rho=eta, xi_rho=xi))
        dout.angle.data = np.angle(angle)
        angle = np.cos(self._obj.angle_xy) + np.sin(self._obj.angle_xy)*1j
        angle = angle.interp(dict(eta_rho=eta, xi_rho=xi))
        dout.angle_xy.data = np.angle(angle)

        # Add extra coordinates
        dout.coords['eta'], dout.coords['xi'] = eta, xi
        dout.coords['lon_vert'] = ds_locs.lon_vert
        dout.coords['lat_vert'] = ds_locs.lat_vert
        dout.coords['x_vert'] = ds_locs.x_vert
        dout.coords['y_vert'] = ds_locs.y_vert
        return dout

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
        ds_locs = self.set_locs(lon, lat, time, get_vert=True)
        dout = self._interp(ds_locs)
        dout.coords['track'] = ds_locs.track
        dout.coords['distance'] = ds_locs.distance

        # Rotate velocities to allign with along/cross directions
        if self._is_dataset:
            geod = pyproj.Geod(ellps='WGS84')
            angle, _, _ = geod.inv(
                ds_locs.lon_vert[:-1], ds_locs.lat_vert[:-1],
                ds_locs.lon_vert[1:], ds_locs.lat_vert[1:])
            angle = (90.-angle)*np.pi/180.
            ar = dout.angle - angle
            dout['angle_rot'] = ar

            if 'u' in dout and 'v' in dout:
                dout['u_rot'] = dout.u*np.cos(ar) - dout.v*np.sin(ar)
                dout['v_rot'] = dout.v*np.cos(ar) + dout.u*np.sin(ar)
            if 'ubar' in dout and 'vbar' in dout:
                dout['ubar_rot'] = dout.ubar*np.cos(ar) - dout.vbar*np.sin(ar)
                dout['vbar_rot'] = dout.vbar*np.cos(ar) + dout.ubar*np.sin(ar)
        return dout

    def interp2d(self, lon, lat, time=None):
        """
        Horizontal interpolation method for ROMS Dataset.

        ds = roms.interp(lon, lat, time=None)
        Inputs:
            lon, lat - 2-D array of longitude/latitude
            time (optional) - 1-D list of time
        Output:
            ds - interpolated Dataset.
        """
        ds_locs = self.set_locs2d(lon, lat, time, get_vert=True)
        ds_locs = ds_locs.stack(track=('yc', 'xc'))
        dout = self._interp(ds_locs)
        dout = dout.assign_coords(track=ds_locs.track)
        dout = dout.unstack('track')
        ds_locs = ds_locs.unstack('track')
        dout.coords['angle_xyr'] = ds_locs.angle_xyr

        # Rotate velocities to allign with along/cross directions
        if self._is_dataset:
            geod = pyproj.Geod(ellps='WGS84')
            angle, _, _ = geod.inv(
                ds_locs.lon_vert[:, :-1], ds_locs.lat_vert[:, :-1],
                ds_locs.lon_vert[:, 1:], ds_locs.lat_vert[:, 1:])
            angle = (90.-angle)*np.pi/180.
            angle = np.cos(angle) + np.sin(angle)*1j
            angle = np.angle(0.5*(angle[1:, :] + angle[:-1, :]))
            ar = dout.angle - angle
            dout['angle_rot'] = ar

            if 'u' in dout and 'v' in dout:
                dout['u_rot'] = dout.u*np.cos(ar) - dout.v*np.sin(ar)
                dout['v_rot'] = dout.v*np.cos(ar) + dout.u*np.sin(ar)
            if 'ubar' in dout and 'vbar' in dout:
                dout['ubar_rot'] = dout.ubar*np.cos(ar) - dout.vbar*np.sin(ar)
                dout['vbar_rot'] = dout.vbar*np.cos(ar) + dout.ubar*np.sin(ar)
        return dout

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

    def isoline(self, xi0, eta0, depth, dis_max, idx_type='lonlat'):
        """
        Get isoline coordinates.
        """
        fig, ax = plt.subplots()
        ct = ax.contour(self.x, self.y, self.h, [depth], colors='k')
        isolines = ct.allsegs[0]
        plt.close(fig)

        if idx_type == 'lonlat':
            proj = pyproj.Proj(self._obj.attrs['proj4_init'])
            x0, y0 = proj(xi0, eta0)
        elif idx_type == 'index':
            x0 = self.x.interp(eta_rho=eta0, xi_rho=xi0)
            y0 = self.x.interp(eta_rho=eta0, xi_rho=xi0)
        elif idx_type == 'xy':
            x0, y0 = xi0, eta0
        else:
            raise ValueError('Unknown index type.')

        dis_min = []
        for i in range(len(isolines)):
            x, y = isolines[i].T
            dis_min.append(np.hypot(x-x0, y-y0).min())
        idx = np.argmin(np.array(dis_min))
        x, y = isolines[idx].T
        in_range = np.where(np.hypot(x-x0, y-y0) <= dis_max)[0]
        idx0, idx1 = in_range[0], in_range[-1]
        return x[idx0:idx1], y[idx0:idx1]

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

    def transform(self, z):
        """
        Vertical coordinate transformer.
        ds = roms.transform(z)
        """
        # TODO
        return

    def uv_rotate(self, uvar='u', vvar='v'):
        """
        Rotate U/V velocity from XI/ETA coordinates to X/Y coordinates
        """
        angle = self._obj.angle_xy
        udims, vdims = self._obj.u.dims, self._obj.v.dims
        if 'xi_rho' in udims and 'eta_rho' in udims and \
           'xi_rho' in vdims and 'eta_rho' in vdims:
            u, v = self._obj[uvar], self._obj[vvar]
        else:
            u = self._obj[uvar].interp(dict(xi_u=self._obj.xi_rho,
                                            eta_u=self._obj.eta_rho))
            v = self._obj[vvar].interp(dict(xi_v=self._obj.xi_rho,
                                            eta_v=self._obj.eta_rho))

        u_x = u*np.cos(angle) - v*np.sin(angle)
        v_y = v*np.cos(angle) + u*np.sin(angle)
        return u_x, v_y

    def zr_to_zw(self, var):
        hpos = _find_hpos(self._obj[var])
        if hpos == '_rho':
            zr, zw = self.z_r_rho, self.z_w_rho
        elif hpos == '_u':
            zr, zw = self.z_r_u, self.z_w_u
        elif hpos == '_v':
            zr, zw = self.z_r_v, self.z_w_v
        elif hpos == '_psi':
            zr, zw = self.z_r_psi, self.z_w_psi

        out = xr.apply_ufunc(
            _interp1d,
            self._obj[var], zr, zw,
            input_core_dims=[['s_rho'], ['s_rho'], ['s_w']],
            output_core_dims=[['s_w']],
            exclude_dims=set(('s_rho',)),
            dask='parallelized')
        return out.transpose(*zw.dims)

    def vgrad(self, var):
        vpos = _find_vpos(self._obj[var])
        hpos = _find_hpos(self._obj[var])
        if vpos == '_r':
            dva = self.zr_to_zw(var).diff(dim='s_w').data
        elif vpos == '_w':
            dva = self._obj[var].diff(dim='s_w').data
        if hpos == '_rho':
            dz = self.dz_rho
        elif hpos == '_u':
            dz = self.dz_u
        elif hpos == '_v':
            dz = self.dz_v
        elif hpos == '_psi':
            dz = self.dz_psi
        ddz = dva/dz
        return ddz

    def hgrad(self, var, direction='both', conserve=True):
        ndim = self._obj[var].ndim
        hpos = _find_hpos(self._obj[var])

        coord_transx = {'_rho': '_u', '_u': '_rho', '_v': '_psi', '_psi': '_v'}
        coord_transy = {'_rho': '_v', '_v': '_rho', '_u': '_psi', '_psi': '_u'}

        fposx, fposy = coord_transx[hpos], coord_transy[hpos]

        fetax, fxix = 'eta' + fposx, 'xi' + fposx
        fetay, fxiy = 'eta' + fposy, 'xi' + fposy

        dimsx = self._obj[var].dims[:-2] + (fetax, fxix)
        dimsy = self._obj[var].dims[:-2] + (fetay, fxiy)

        coordsx, coordsy = {}, {}
        for ix, iy in zip(dimsx, dimsy):
            coordsx[ix], coordsy[iy] = self._obj[ix], self._obj[iy]

        if hpos == '_rho':
            pm, pn = self._obj.pm.data, self._obj.pn.data
        else:
            eta, xi = self._obj['eta' + hpos], self._obj['xi' + hpos]
            pm = self._obj.pm.interp(dict(eta_rho=eta, xi_rho=xi)).data
            pn = self._obj.pn.interp(dict(eta_rho=eta, xi_rho=xi)).data

        va = self._obj[var].data

        slc, pad_width = (), ()
        for i in range(ndim-2):
            slc = slc + (slice(None),)
            pad_width = pad_width + ((0, 0),)
        slcx0 = (slice(None), slice(1, None))
        slcx1 = (slice(None), slice(None, -1))
        slcy0 = (slice(1, None), slice(None))
        slcy1 = (slice(None, -1), slice(None))
        pad_widthx = pad_width + ((0, 0), (1, 1))
        pad_widthy = pad_width + ((1, 1), (0, 0))

        if conserve:
            facx = 0.25 * (pn[:, 1:] + pn[:, :-1]) * (pm[:, 1:] + pm[:, :-1])
            facy = 0.25 * (pm[1:, :] + pm[:-1, :]) * (pn[1:, :] + pn[:-1, :])
            if direction in ['both', 'x']:
                ddx = (va[slc + slcx0]/pn[slcx0]-va[slc + slcx1]/pn[slcx1]) * \
                      facx
            if direction in ['both', 'y']:
                ddy = (va[slc + slcy0]/pm[slcy0]-va[slc + slcy1]/pm[slcy1]) * \
                      facy
        else:
            if direction in ['both', 'x']:
                ddx = (va[slc + slcx0]-va[slc + slcx1]) * \
                      0.5*(pm[slcx0]+pm[slcx1])
            if direction in ['both', 'y']:
                ddy = (va[slc + slcy0]-va[slc + slcy1]) * \
                      0.5*(pn[slcy0]+pn[slcy1])

        if direction in ['both', 'x']:
            if hpos in ['_u', '_psi']:
                ddx = np.pad(ddx, pad_widthx, constant_values=np.nan)
            ddx = DataArray(data=ddx, dims=dimsx, coords=coordsx,
                            attrs=dict(long_name='XI-direction gradient'))
        if direction in ['both', 'y']:
            if hpos in ['_v', '_psi']:
                ddy = np.pad(ddy, pad_widthy, constant_values=np.nan)
            ddy = DataArray(data=ddy, dims=dimsy, coords=coordsy,
                            attrs=dict(long_name='ETA-direction gradient'))

        if direction == 'both':
            return ddx, ddy
        elif direction == 'x':
            return ddx
        elif direction == 'y':
            return ddy
        else:
            return

    def hgrad3d(self, var, direction='both'):
        vpos = _find_vpos(self._obj[var])
        hpos = _find_hpos(self._obj[var])

        dvdx, dvdy = self.hgrad(var, 'both')
        if vpos == '_w':
            dvdx = dvdx.interp(s_w=self._obj.s_rho)
            dvdy = dvdy.interp(s_w=self._obj.s_rho)

        if hpos == '_rho':
            _ = self.z_r_rho
        elif hpos == '_u':
            _ = self.z_r_u
        elif hpos == '_v':
            _ = self.z_r_v
        elif hpos == '_psi':
            _ = self.z_r_psi
        dzdx, dzdy = self.hgrad('_z_r' + hpos)

        dvdz = self.vgrad(var)
        coord_transx = {'_rho': '_u', '_u': '_rho', '_v': '_psi', '_psi': '_v'}
        coord_transy = {'_rho': '_v', '_v': '_rho', '_u': '_psi', '_psi': '_u'}
        dvdzx = dvdz.interp(
            {'xi' + hpos: self._obj['xi' + coord_transx[hpos]],
             'eta' + hpos: self._obj['eta' + coord_transx[hpos]]})
        dvdzy = dvdz.interp(
            {'xi' + hpos: self._obj['xi' + coord_transy[hpos]],
             'eta' + hpos: self._obj['eta' + coord_transy[hpos]]})

        ddx = dvdx - dzdx*dvdzx
        ddy = dvdy - dzdy*dvdzy

        if direction == 'both':
            return ddx, ddy
        elif direction == 'x':
            return ddx
        elif direction == 'y':
            return ddy
        else:
            return

    def hgrad_x(self, var):
        coord_trans = {'_rho': '_u', '_u': '_rho', '_v': '_psi', '_psi': '_v'}

        ndim = self._obj[var].ndim
        hpos = _find_hpos(self._obj[var])
        fpos = coord_trans[hpos]
        feta, fxi = 'eta' + fpos, 'xi' + fpos
        dims = self._obj[var].dims[:-2] + (feta, fxi)
        coords = {}
        for idim in dims:
            coords[idim] = self._obj[idim]

        if hpos == '_rho':
            pm, pn = self._obj.pm.data, self._obj.pn.data
        else:
            eta, xi = self._obj['eta' + hpos], self._obj['xi' + hpos]
            pm = self._obj.pm.interp(dict(eta_rho=eta, xi_rho=xi)).data
            pn = self._obj.pn.interp(dict(eta_rho=eta, xi_rho=xi)).data
        fac = 0.25 * (pn[:, 1:] + pn[:, :-1]) * (pm[:, 1:] + pm[:, :-1])

        va = self._obj[var].data
        if ndim == 2:
            ddx = (va[:, 1:]/pn[:, 1:]-va[:, :-1]/pn[:, :-1])*fac
            pad_width = ((0, 0), (1, 1))
        elif ndim == 3:
            ddx = (va[:, :, 1:]/pn[:, 1:]-va[:, :, :-1]/pn[:, :-1])*fac
            pad_width = ((0, 0), (0, 0), (1, 1))
        elif ndim == 4:
            ddx = (va[:, :, :, 1:]/pn[:, 1:]-va[:, :, :, :-1]/pn[:, :-1])*fac
            pad_width = ((0, 0), (0, 0), (0, 0), (1, 1))

        if hpos in ['_u', '_psi']:
            ddx = np.pad(ddx, pad_width, constant_values=np.nan)

        ddx = DataArray(data=ddx, dims=dims, coords=coords,
                        attrs=dict(long_name='XI-direction gradient'))
        return ddx

    def hgrad_y(self, var):
        coord_trans = {'_rho': '_v', '_v': '_rho', '_u': '_psi', '_psi': '_u'}

        ndim = self._obj[var].ndim
        hpos = _find_hpos(self._obj[var])
        fpos = coord_trans[hpos]
        feta, fxi = 'eta' + fpos, 'xi' + fpos
        dims = self._obj[var].dims[:-2] + (feta, fxi)
        coords = {}
        for idim in dims:
            coords[idim] = self._obj[idim]

        if hpos == '_rho':
            pm, pn = self._obj.pm.data, self._obj.pn.data
        else:
            eta, xi = self._obj['eta' + hpos], self._obj['xi' + hpos]
            pm = self._obj.pm.interp(dict(eta_rho=eta, xi_rho=xi)).data
            pn = self._obj.pn.interp(dict(eta_rho=eta, xi_rho=xi)).data
        fac = 0.25 * (pm[1:, :] + pm[:-1, :]) * (pn[1:, :] + pn[:-1, :])

        va = self._obj[var].data
        if ndim == 2:
            ddy = (va[1:, :]/pm[1:, :]-va[:-1, :]/pm[:-1, :])*fac
            pad_width = ((1, 1), (0, 0))
        elif ndim == 3:
            ddy = (va[:, 1:, :]/pm[1:, :]-va[:, :-1, :]/pm[:-1, :])*fac
            pad_width = ((0, 0), (1, 1), (0, 0))
        elif ndim == 4:
            ddy = (va[:, :, 1:, :]/pm[1:, :]-va[:, :, :-1, :]/pm[:-1, :])*fac
            pad_width = ((0, 0), (0, 0), (1, 1), (0, 0))

        if hpos in ['_v', '_psi']:
            ddy = np.pad(ddy, pad_width, constant_values=np.nan)

        ddy = DataArray(data=ddy, dims=dims, coords=coords,
                        attrs=dict(long_name='ETA-direction gradient'))
        return ddy

    def vorticity(self, wind=False):
        """
        Calcuate relative_vorticity.
        """
        pm, pn = self._obj.pm.data, self._obj.pn.data
        if wind:
            u, v = self._obj.sustr.data, self._obj.svstr.data
        else:
            u, v = self._obj.ubar.data, self._obj.vbar.data
        ndim = u.ndim
        dx_u = 2./(pm[:, 1:] + pm[:, :-1])
        dy_v = 2./(pn[1:, :] + pn[:-1, :])
        if ndim == 3:
            dude = dx_u[1:, :]*u[:, 1:, :] - dx_u[:-1, :]*u[:, :-1, :]
            dvdx = dy_v[:, 1:]*v[:, :, 1:] - dy_v[:, :-1]*v[:, :, :-1]
        else:
            dude = dx_u[1:, :]*u[1:, :] - dx_u[:-1, :]*u[:-1, :]
            dvdx = dy_v[:, 1:]*v[:, 1:] - dy_v[:, :-1]*v[:, :-1]
        rvor = 0.0625 * \
            (pm[1:, 1:] + pm[1:, :-1] + pm[:-1, 1:] + pm[:-1, :-1]) * \
            (pn[1:, 1:] + pn[1:, :-1] + pn[:-1, 1:] + pn[:-1, :-1]) * \
            (dvdx - dude)
        if ndim == 3:
            self._obj['vorticity'] = DataArray(
                    data=rvor,
                    dims=['ocean_time', 'eta_psi', 'xi_psi'],
                    attrs=dict(long_name='Vorticity',
                               units='s-1'))
        return self._obj.vorticity

    def pressure(self):
        """
        Caculate pressure.
        """
        g = 9.81
        rho0 = 1025.
        drhodz = 0.00478
        GRho = g/rho0
        HalfGRho = 0.5*GRho
        rho = self._obj.rho + 1000.
        z_r = self.z_r
        z_w = self.z_w
        rho = rho.transpose('s_rho', ...)
        z_r = z_r.transpose('s_rho', ...)
        z_w = z_w.transpose('s_w', ...)
        dims = rho.dims
        coords = rho.coords
        rho, z_r, z_w = rho.data, z_r.data, z_w.data
        if 'zice' in self._obj:
            zice = self._obj.zice.data
        else:
            zice = 0.

        dR = np.zeros(z_w.shape)
        dZ = np.zeros(z_w.shape)
        dR[1:-1] = rho[1:] - rho[:-1]
        dZ[1:-1] = z_r[1:] - z_r[:-1]
        dR[-1] = dR[-2]
        dZ[-1] = dZ[-2]
        dR[0] = dR[1]
        dZ[0] = dZ[1]
        cff0 = 2*dR[1:]*dR[:-1]
        cff = np.zeros(cff0.shape)
        cff = cff0 / (dR[1:] + dR[:-1])
        cff[cff0 <= 1.e-20] = 0.
        dR[1:] = cff
        cff = 2*dZ[1:]*dZ[:-1] / (dZ[1:] + dZ[:-1])
        dZ[1:] = cff

        cff1 = 1. / (z_r[-1] - z_r[-2])
        cff2 = 0.5 * (rho[-1] - rho[-2]) * \
                     (z_w[-1] - z_r[-1])*cff1
        ptop = g*(z_w[-1] + zice) + \
            GRho*(rho[-1] - 0.5*drhodz*zice)*zice + \
            GRho*(rho[-1] + cff2)*(z_w[-1] - z_r[-1])
        prs = np.zeros(rho.shape)
        prs[-1] = ptop
        for i in range(prs.shape[0]-2, -1, -1):
            delta_prs = HalfGRho * \
                ((rho[i+1] + rho[i])*(z_r[i+1] - z_r[i]) -
                 0.2*((dR[i+1] - dR[i]) *
                      (z_r[i+1] - z_r[i] - 1./12.*(dZ[i+1]+dZ[i])) -
                      (dZ[i+1] - dZ[i]) *
                      (rho[i+1] - rho[i] - 1./12.*(dR[i+1]+dR[i]))
                      )
                 )
            prs[i] = prs[i+1] + delta_prs
        prs = DataArray(data=prs, dims=dims, coords=coords)
        prs = prs.transpose(*self._obj.rho.dims)
        prs.attrs = dict(long_name='pressure', units='Pa')
        return prs*rho0

    def pressure_bot(self):
        """
        Caculate bottom pressure.
        """
        g = 9.81
        drdz = 0.00478
        rho = self._obj.rho + 1000.
        bprs = (rho*g*self.dz).sum(dim='s_rho')
        if 'zice' in self._obj:
            bprs = bprs + \
                   (rho.isel(s_rho=-1)-0.5*drdz*self._obj.zice) * g * \
                   self._obj.zice
        bprs.attrs = dict(long_name='Bottom pressure', units='Pa')
        return bprs

    def potential_energy(self):
        """
        Caculate potential_energy.
        """
        g = 9.81
        drdz = 0.00478
        rho0 = 1025.
        rho = self._obj.rho + 1000.
        z_r = self.z_r
        dz = self.dz
        pote = (rho*g*dz*z_r).sum(dim='s_rho')
        if 'zice' in self._obj:
            pote = pote + g * \
                   (rho.isel(s_rho=-1)*(-0.5*self._obj.zice**2) -
                    0.5*drdz*(1./3.)*self._obj.zice**3)
        pote = pote/rho0
        pote.attrs = dict(long_name='Potential energy', units='m3s-2')
        return pote

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

    def z_r_psi(self):
        if '_z_r_psi' not in self._obj.coords:
            self._obj.coords['_z_r_psi'] = _calc_z(self._obj, '_r', '_psi')
        return self._obj.coords['_z_r_psi']

    @property
    def z_w_psi(self):
        if '_z_w_psi' not in self._obj.coords:
            self._obj.coords['_z_w_psi'] = _calc_z(self._obj, '_w', '_psi')
        return self._obj.coords['_z_w_psi']

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
    def dz_psi(self):
        if '_dz_psi' not in self._obj:
            _dz = self.z_w_psi.diff(dim='s_w')
            _dz = _dz.rename(s_w='s_rho').assign_coords(
                s_rho=self._obj.s_rho)
            self._obj['_dz_psi'] = _dz
        return self._obj['_dz_psi']

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
    hpos = property(lambda self: self._hpos[0])
    vpos = property(lambda self: self._vpos[0])

    @property
    def s_nam(self):
        if self._vpos[0] == '_r':
            return 's_rho'
        else:
            return 's' + self._vpos[0]
    xi_nam = property(lambda self: 'xi' + self._hpos[0])
    eta_nam = property(lambda self: 'eta' + self._hpos[0])

    s = property(lambda self: self._obj[self.s_nam])
    xi = property(lambda self: self._obj[self.xi_nam])
    eta = property(lambda self: self._obj[self.eta_nam])
    x = property(lambda self: self._obj['x' + self._hpos[0]])
    y = property(lambda self: self._obj['y' + self._hpos[0]])
    lon = property(lambda self: self._obj['lon' + self._hpos[0]])
    lat = property(lambda self: self._obj['lat' + self._hpos[0]])
    h = property(lambda self: self._obj['h' + self._hpos[0]])
    mask = property(lambda self: self._obj['mask' + self._hpos[0]])

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

    def transform(self, z):
        """
        Vertical coordinate transformer (with XGCM).

        da = roms.transform(z)
        """
        if type(z) in [int, float]:
            z = np.array([float(z)])
        else:
            z = np.asarray(z)
        z = DataArray(-np.abs(z), dims='Z')
        ds = self.z.to_dataset(name='z')
        grd = Grid(ds, coords={'S': {'center': self.s_nam}},
                   periodic=False)
        da = grd.transform(self._obj, 'S', z, target_data=self.z)
        dims = [i if i != self.s_nam else 'Z' for i in self._obj.dims]
        da = da.transpose(*dims)
        da = da.assign_coords(Z=z)
        return da

    def interpz(self, z, zdim):
        zraw = self.z

        dims = [i for i in z.dims if i != zdim]
        dimsraw = [i for i in zraw.dims if i != self.s_nam]
        assert dims == dimsraw, \
            'z must have the same dimensions as the raw DataArray, ' + \
            'excluding the z-dimension.'

        out = xr.apply_ufunc(
            _interp1d,
            self._obj, zraw, z,
            input_core_dims=[[self.s_nam], [self.s_nam], [zdim]],
            output_core_dims=[[zdim]],
            exclude_dims=set((self.s_nam,)),
            dask='parallelized')
        return out.transpose(*z.dims)

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


def open_mfdataset(filename, grid_filename=None, interp_rho=False, **kwargs):
    """
    Open a ROMS history/average file, optionally a ROMS grid file, and
    construct a RDataset.

    ds = open_mfdataset(filename, grid_filename=None, interp_rho=False,
                        **kwargs)
    Inputs:
        filename      - str, ROMS file path
        grid_filename - str, ROMS grid file path
        interp_rho    - bool, if to interpolate all dataArray to rho points
        **kwargs      - other keyword arguments to be passed to RDataset
    """
    ds = xr.open_mfdataset(filename, **kwargs)
    if grid_filename is not None:
        if isinstance(filename, list):
            grd = grid.get_ROMS_grid(grid_file=grid_filename,
                                     hist_file=filename[0])
        else:
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
        self.coords['eta_rho'] = np.arange(self.dims['eta_rho']) + 0.5
        self.coords['xi_rho'] = np.arange(self.dims['xi_rho']) + 0.5
        self.coords['eta_vert'] = np.arange(self.dims['eta_rho']+1)
        self.coords['xi_vert'] = np.arange(self.dims['xi_rho']+1)
        self.coords['eta_psi'] = np.arange(self.dims['eta_psi']) + 1.0
        self.coords['xi_psi'] = np.arange(self.dims['xi_psi']) + 1.0
        self.coords['eta_u'] = self.eta_rho.data.copy()
        self.coords['xi_u'] = self.xi_psi.data.copy()
        self.coords['eta_v'] = self.eta_psi.data.copy()
        self.coords['xi_v'] = self.xi_rho.data.copy()

        # ---------------------------------------------------------------
        # Perform interpolation to RHO points or generate coords for u/v
        # points
        if self._interp_rho:
            # Interpolate and remove u/v related coords
            uv_vars = []
            for var in self.data_vars:
                vdims = self[var].dims
                if ('xi_u' in vdims or 'eta_u' in vdims or
                    'xi_v' in vdims or 'eta_v' in vdims) and \
                   'ocean_time' in vdims:
                    uv_vars.append(var)
                    self[var] = self[var].where(~self[var].isnull(), other=0)
            itp = self.interp(eta_u=self.eta_rho, eta_v=self.eta_rho,
                              xi_u=self.xi_rho, xi_v=self.xi_rho,
                              s_w=self.s_rho)
            for var in self.data_vars:
                self[var] = itp[var]
            for var in uv_vars:
                self[var] = self[var].where(self.mask_rho == 1)
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
            self['h_psi'] = self.h.interp(
                eta_rho=self.eta_psi, xi_rho=self.xi_psi)
            if 'zeta' in self:
                self['zeta_u'] = self.zeta.interp(
                    eta_rho=self.eta_u, xi_rho=self.xi_u)
                self['zeta_v'] = self.zeta.interp(
                    eta_rho=self.eta_v, xi_rho=self.xi_v)
                self['zeta_psi'] = self.zeta.interp(
                    eta_rho=self.eta_psi, xi_rho=self.xi_psi)
            else:
                self['zeta'] = xr.zeros_like(self.h)
                self['zeta_u'] = xr.zeros_like(self.h_u)
                self['zeta_v'] = xr.zeros_like(self.h_v)
                self['zeta_psi'] = xr.zeros_like(self.h_psi)
            if 'zice' in self:
                self['zice_u'] = self.zice.interp(
                    eta_rho=self.eta_u, xi_rho=self.xi_u)
                self['zice_v'] = self.zice.interp(
                    eta_rho=self.eta_v, xi_rho=self.xi_v)
                self['zice_psi'] = self.zice.interp(
                    eta_rho=self.eta_psi, xi_rho=self.xi_psi)
            else:
                self['zice'] = xr.zeros_like(self.h)
                self['zice_u'] = xr.zeros_like(self.h_u)
                self['zice_v'] = xr.zeros_like(self.h_v)
                self['zice_psi'] = xr.zeros_like(self.h_psi)

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
        if self._spherical:
            proj4_init = self._grid.hgrid.proj.to_proj4()
        else:
            proj4_init = None
        self.attrs['proj4_init'] = proj4_init
        for var in self:
            self[var].attrs['proj4_init'] = proj4_init

        # When Dataset is loaded with open_mfdataset, clean some variables
        # by averaging over ocean_time.
        if ('ntimes' in self) and (self.ntimes.dims == ('ocean_time', )):
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
                self.coords['zeta_psi'] = self.zeta_psi

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
