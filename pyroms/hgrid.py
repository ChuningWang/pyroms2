"""
Tools for creating and working with ROMS Arakawa C-Grids.
"""

# python Typing functionalities
from typing import List, Tuple, Union

# Python/C interface
import os
import glob
import ctypes

# computation kernels
import numpy as np
import pyroms

# projection utilities
import pyproj
import shapely.ops as sops
import shapely.geometry as sgeometry

# plotting tools
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.artist import Artist

# map plotting tools
import cartopy.crs as ccrs
import cartopy.io as cio


# Python Type utilities
_atype = Union[type(None), np.ndarray]
_ftype = Union[type(None), float]
_fatype = Union[float, np.ndarray]


class CGrid:
    """
    Curvilinear Arakawa C-Grid

    The basis for the CGrid class are two arrays defining the verticies of the
    grid in Cartesian (for geographic coordinates, see CGridGeo). An optional
    mask may be defined on the cell centers. Other Arakawa C-grid properties,
    such as the locations of the cell centers (rho-points), cell edges (u and
    v velocity points), cell widths (dx and dy) and other metrics (angle,
    dmde, and dndx) are all calculated internally from the vertex points.

    Input vertex arrays may be either type np.array or np.ma.MaskedArray. If
    masked arrays are used, the mask will be a combination of the specified
    mask (if given) and the masked locations.

    EXAMPLES:
    --------

    >>> x, y = mgrid[0.0:7.0, 0.0:8.0]
    >>> x = np.ma.masked_where( (x<3) & (y<3), x)
    >>> y = np.ma.MaskedArray(y, x.mask)
    >>> grd = pyroms.grid.CGrid(x, y)
    >>> print(grd.x_rho)
    [[ --  --  -- 0.5 0.5 0.5 0.5]
     [ --  --  -- 1.5 1.5 1.5 1.5]
     [ --  --  -- 2.5 2.5 2.5 2.5]
     [3.5 3.5 3.5 3.5 3.5 3.5 3.5]
     [4.5 4.5 4.5 4.5 4.5 4.5 4.5]
     [5.5 5.5 5.5 5.5 5.5 5.5 5.5]]
    >>> print(grd.mask)
    [[ 0.  0.  0.  1.  1.  1.  1.]
     [ 0.  0.  0.  1.  1.  1.  1.]
     [ 0.  0.  0.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.]
     [ 1.  1.  1.  1.  1.  1.  1.]]
    """

    def __init__(self,
                 x_vert: np.ndarray, y_vert: np.ndarray,
                 x_rho: _atype = None, y_rho: _atype = None,
                 x_psi: _atype = None, y_psi: _atype = None,
                 x_u: _atype = None, y_u: _atype = None,
                 x_v: _atype = None, y_v: _atype = None,
                 dx: _atype = None, dy: _atype = None,
                 dndx: _atype = None, dmde: _atype = None,
                 angle_rho: _atype = None, mask_rho: _atype = None):

        assert x_vert.ndim == 2 and y_vert.ndim == 2 and \
               x_vert.shape == y_vert.shape, \
               'x and y must be 2D arrays of the same size.'

        if np.any(np.isnan(x_vert)) or np.any(np.isnan(y_vert)):
            x_vert = np.ma.masked_where(
                (np.isnan(x_vert)) | (np.isnan(y_vert)), x_vert)
            y_vert = np.ma.masked_where(
                (np.isnan(x_vert)) | (np.isnan(y_vert)), y_vert)

        # Pass in x/y of vertices
        self._x_vert, self._y_vert = x_vert, y_vert

        # Initiallize geographic Parameter (even not used in CGrid)
        self.f, self.spherical, self.proj = None, None, None

        # Set land masks
        rho_shape = tuple([n-1 for n in self.x_vert.shape])
        if mask_rho is None:
            mask_rho = np.ones(rho_shape)
        else:
            assert mask_rho.shape == rho_shape, \
                'Mask size does not match mesh size.'

        # If maskedarray is given for verticies, modify the mask such that
        # non-existant grid points are masked. A cell requires all four
        # verticies to be defined as a water point.
        if isinstance(self._x_vert, np.ma.MaskedArray):
            mask = (self._x_vert.mask[:-1, :-1] | self._x_vert.mask[1:, :-1] |
                    self._x_vert.mask[:-1, 1:] | self._x_vert.mask[1:, 1:])*1
            mask_rho = mask_rho*mask
        if isinstance(self._y_vert, np.ma.MaskedArray):
            mask = (self._y_vert.mask[:-1, :-1] | self._y_vert.mask[1:, :-1] |
                    self._y_vert.mask[:-1, 1:] | self._y_vert.mask[1:, 1:])*1
            mask_rho = mask_rho*mask

        # Set mask to integer type to save memory
        self.mask_rho = mask_rho.astype(np.int32)

        # If grid rho/u/v/psi coordinates not given, calculate coordinates.
        if x_rho is None or y_rho is None or \
           x_psi is None or y_psi is None or \
           x_u is None or y_u is None or \
           x_v is None or y_v is None:
            self._calculate_subgrids()
        else:
            self.x_rho, self.y_rho = x_rho, y_rho
            self.x_psi, self.y_psi = x_psi, y_psi
            self.x_u, self.y_u = x_u, y_u
            self.x_v, self.y_v = x_v, y_v

        # If grid metrics not given, calculate metrics.
        if dx is None or dy is None:
            self._calculate_metrics()
        else:
            self.dx, self.dy = dx, dy
        self.pm, self.pn = 1./self.dx, 1./self.dy
        self.xl = max(self.dx[0, :].sum(), self.dx[-1, :].sum())
        self.el = max(self.dy[:, 0].sum(), self.dy[:, -1].sum())

        if dndx is None or dmde is None:
            self._calculate_derivative_metrics()
        else:
            self.dndx, self.dmde = dndx, dmde

        # If grid angle (from x-positive) not given, calculate grid angle.
        if angle_rho is None:
            self._calculate_angle_rho()
        else:
            self.angle_rho = angle_rho
        self._calculate_angle_vert()
        return

    def _calculate_subgrids(self):
        """
        Calculate rho/u/v/psi grid coordinates
        """
        self.x_rho = 0.25*(self.x_vert[1:, 1:]+self.x_vert[1:, :-1] +
                           self.x_vert[:-1, 1:]+self.x_vert[:-1, :-1])
        self.y_rho = 0.25*(self.y_vert[1:, 1:]+self.y_vert[1:, :-1] +
                           self.y_vert[:-1, 1:]+self.y_vert[:-1, :-1])
        self.x_psi = self.x_vert[1:-1, 1:-1]
        self.y_psi = self.y_vert[1:-1, 1:-1]
        self.x_u = 0.5*(self.x_vert[:-1, 1:-1] + self.x_vert[1:, 1:-1])
        self.y_u = 0.5*(self.y_vert[:-1, 1:-1] + self.y_vert[1:, 1:-1])
        self.x_v = 0.5*(self.x_vert[1:-1, :-1] + self.x_vert[1:-1, 1:])
        self.y_v = 0.5*(self.y_vert[1:-1, :-1] + self.y_vert[1:-1, 1:])
        return

    def _calculate_metrics(self):
        """
        Calculate grid length dx/dy
        """
        x_temp = 0.5*(self.x_vert[1:, :]+self.x_vert[:-1, :])
        y_temp = 0.5*(self.y_vert[1:, :]+self.y_vert[:-1, :])
        self.dx = np.sqrt(np.diff(x_temp, axis=1)**2 +
                          np.diff(y_temp, axis=1)**2)
        x_temp = 0.5*(self.x_vert[:, 1:]+self.x_vert[:, :-1])
        y_temp = 0.5*(self.y_vert[:, 1:]+self.y_vert[:, :-1])
        self.dy = np.sqrt(np.diff(x_temp, axis=0)**2 +
                          np.diff(y_temp, axis=0)**2)
        return

    def _calculate_derivative_metrics(self):
        """
        Calculate dy/dxi and dx/deta
        """
        self.dndx = self.dy * 0.
        self.dmde = self.dx * 0.
        self.dndx[1:-1, 1:-1] = 0.5*(self.dy[1:-1, 2:] - self.dy[1:-1, :-2])
        self.dmde[1:-1, 1:-1] = 0.5*(self.dx[2:, 1:-1] - self.dx[:-2, 1:-1])
        return

    def _calculate_angle_rho(self):
        """
        Calculate grid angle at rho points
        """
        self.angle_rho = np.arctan2(
            np.diff(0.5*(self.y_vert[1:, :]+self.y_vert[:-1, :])),
            np.diff(0.5*(self.x_vert[1:, :]+self.x_vert[:-1, :])))
        return

    def _calculate_angle_vert(self):
        """
        Calculate grid angle at vert points
        """
        angle_ud = np.arctan2(np.diff(self.y_vert, axis=1),
                              np.diff(self.x_vert, axis=1))
        angle_lr = np.arctan2(np.diff(self.y_vert, axis=0),
                              np.diff(self.x_vert, axis=0)) - np.pi/2.0
        self.angle = self.x_vert * self.y_vert * 0.

        # domain center
        self.angle[1:-1, 1:-1] = 0.25*(angle_ud[1:-1, 1:]+angle_ud[1:-1, :-1] +
                                       angle_lr[1:, 1:-1]+angle_lr[:-1, 1:-1])
        # edges
        self.angle[0, 1:-1] = (1.0/3.0) * \
            (angle_lr[0, 1:-1]+angle_ud[0, 1:]+angle_ud[0, :-1])
        self.angle[-1, 1:-1] = (1.0/3.0) * \
            (angle_lr[-1, 1:-1]+angle_ud[-1, 1:]+angle_ud[-1, :-1])
        self.angle[1:-1, 0] = (1.0/3.0) * \
            (angle_ud[1:-1, 0]+angle_lr[1:, 0]+angle_lr[:-1, 0])
        self.angle[1:-1, -1] = (1.0/3.0) * \
            (angle_ud[1:-1, -1]+angle_lr[1:, -1]+angle_lr[:-1, -1])

        # conrers
        self.angle[0, 0] = 0.5*(angle_lr[0, 0]+angle_ud[0, 0])
        self.angle[0, -1] = 0.5*(angle_lr[0, -1]+angle_ud[0, -1])
        self.angle[-1, 0] = 0.5*(angle_lr[-1, 0]+angle_ud[-1, 0])
        self.angle[-1, -1] = 0.5*(angle_lr[-1, -1]+angle_ud[-1, -1])
        return

    def calculate_orthogonality(self):
        """
        Calculate orthogonality error in radians
        """
        z = self.x_vert + 1j*self.y_vert
        du = np.diff(z, axis=1)
        du = (du/abs(du))[:-1, :]
        dv = np.diff(z, axis=0)
        dv = (dv/abs(dv))[:, :-1]
        ang1 = np.arccos(du.real*dv.real + du.imag*dv.imag)
        du = np.diff(z, axis=1)
        du = (du/abs(du))[1:, :]
        dv = np.diff(z, axis=0)
        dv = (dv/abs(dv))[:, :-1]
        ang2 = np.arccos(du.real*dv.real + du.imag*dv.imag)
        du = np.diff(z, axis=1)
        du = (du/abs(du))[:-1, :]
        dv = np.diff(z, axis=0)
        dv = (dv/abs(dv))[:, 1:]
        ang3 = np.arccos(du.real*dv.real + du.imag*dv.imag)
        du = np.diff(z, axis=1)
        du = (du/abs(du))[1:, :]
        dv = np.diff(z, axis=0)
        dv = (dv/abs(dv))[:, 1:]
        ang4 = np.arccos(du.real*dv.real + du.imag*dv.imag)
        ang = np.mean([abs(ang1), abs(ang2), abs(ang3), abs(ang4)], axis=0)
        ang = (ang-np.pi/2.0)

        self.angle_orthogonality = ang
        return ang

    def mask_polygon(self,
                     polyverts: np.ndarray,
                     mask_value: int = 0,
                     use_iceshelf: bool = False):
        """
        Mask Cartesian points contained within the polygon defined by polyverts

        A cell is masked if the cell center (x_rho, y_rho) is within the
        polygon. Other sub-masks (mask_u, mask_v, and mask_psi) are updated
        automatically.

        mask_value [=0.0] may be specified to alter the value of the mask set
        within the polygon.  E.g., mask_value=1 for water points.
        """

        assert polyverts.ndim == 2, \
            'polyverts must be a 2D array, or a similar sequence'
        assert polyverts.shape[1] == 2, \
            'polyverts must be two columns of points'
        assert polyverts.shape[0] > 2, \
            'polyverts must contain at least 3 points'

        path = Path(polyverts)
        inside = path.contains_points(
            np.vstack((self.x_rho.flatten(), self.y_rho.flatten())).T)
        if np.any(inside):
            if use_iceshelf:
                self.mask_is.flat[inside] = mask_value
            else:
                self.mask_rho.flat[inside] = mask_value
        return

    def mask_padding(self):
        """
        This funciton fixes small issues of grid mask, for example, if an
        ocean grid is surrounded by three land grid, it is difficult to
        advect energy in/out of the grid and thus should be masked.

        This fixer, of cause if very crude and cannot remove all
        questionable points. The function edit_mask() or edit_mask_ij()
        should be used to manually fix other issues in the grid.
        """

        mask = self.mask_rho.copy()
        for ci in range(1000):
            mask_ct = np.zeros(mask.shape)
            mask_ct[1:-1, 1:-1] = \
                mask[1:-1, 2:] + mask[1:-1, :-2] + \
                mask[2:, 1:-1] + mask[:-2, 1:-1]
            maskl = (mask_ct <= 1) & (mask == 1)
            if np.any(maskl):
                mask[maskl] = 0
            else:
                break
        if ci == 999:
            raise Warning('pyroms: reached maximum loop count (1000) in ' +
                          'mask_padder().')
        self.mask_rho[1:-1, 1:-1] = mask[1:-1, 1:-1]
        return

    def edit_mask_ij(self, coast=None, **kwargs):
        if 'iceshelf' in kwargs.keys():
            iceshelf = kwargs.pop('iceshelf')
        else:
            iceshelf = None
        emi = EditMaskIJ(self, coast=coast, iceshelf=iceshelf)
        emi(**kwargs)
        return

    def edit_mask(self, **kwargs):
        if 'iceshelf' in kwargs.keys():
            iceshelf = kwargs.pop('iceshelf')
        else:
            iceshelf = None
        em = EditMask(self, iceshelf=iceshelf)
        em(**kwargs)
        return

    def get_position(self, **kwargs):
        if 'iceshelf' in kwargs.keys():
            iceshelf = kwargs.pop('iceshelf')
        else:
            iceshelf = None
        self.get_pos = GetPositionFromMap(
            self, proj=self.proj, iceshelf=iceshelf)
        self.get_pos()
        sta_hgrd = pyroms.sta_hgrid.StaHGridGeo(
            self.get_pos.lon, self.get_pos.lat,
            self.get_pos.x, self.get_pos.y,
            self.get_pos.angle, self.proj)
        return sta_hgrd

    def add_focus(self, x0: float, y0: float,
                  xfac: _ftype = None, yfac: _ftype = None,
                  rx: _ftype = 0.1, ry: _ftype = 0.1):
        """
        Add local intensification for CGrid. This function will
        regenerate a new set of CGrid.

        x0, y0     - location to focus on in x/y coordinates.
        xfac, yfac - intensification factors in xi/eta direction.
        rx, ry     - normalized range of intensification.
        """
        self._refocus(x0, y0, xfac, yfac, rx, ry)
        self.x_vert, self.y_vert = self.Gridgen.x, self.Gridgen.y
        return

    def _refocus(self, x0, y0, xfac, yfac, rx, ry):
        # Fine the index of points to be focused
        dis = np.sqrt((self.x_vert - x0)**2 + (self.y_vert - y0)**2)
        yloc, xloc = np.where(dis == dis.min())
        yloc, xloc = yloc[0], xloc[0]

        # Construct the focus function
        foc = Focus(xloc/(self.Gridgen.nx), yloc/(self.Gridgen.ny),
                    xfactor=xfac, yfactor=yfac, rx=rx, ry=ry)

        # Regenerate grid using Gridgen
        self.Gridgen = Gridgen(self.Gridgen.xbry, self.Gridgen.ybry,
                               self.Gridgen.beta,
                               (self.Gridgen.ny, self.Gridgen.nx),
                               focus=foc)
        return

    # property decorators
    x = property(lambda self: self.x_vert)
    y = property(lambda self: self.y_vert)

    @property
    def x_vert(self):
        return self._x_vert

    @x_vert.setter
    def x_vert(self, x_vert):
        self.__init__(x_vert, self._y_vert)

    @property
    def y_vert(self):
        return self._y_vert

    @y_vert.setter
    def y_vert(self, y_vert):
        self.__init__(self._x_vert, y_vert)

    @property
    def mask_rho(self):
        return self._mask_rho

    @mask_rho.setter
    def mask_rho(self, mask_rho):
        self._mask_rho = mask_rho
        self.mask_u = self._mask_rho[:, 1:]*self._mask_rho[:, :-1]
        self.mask_v = self._mask_rho[1:, :]*self._mask_rho[:-1, :]
        self.mask_psi = \
            self._mask_rho[1:, 1:]*self._mask_rho[:-1, 1:] * \
            self._mask_rho[1:, :-1]*self._mask_rho[:-1, :-1]
        self.mask_vert = np.ones(self.x_vert.shape, dtype=np.int32)
        for sli in [(slice(-1), slice(1, None)), (slice(1, None), slice(-1)),
                    (slice(-1), slice(-1)), (slice(1, None), slice(1, None))]:
            self.mask_vert[sli] *= self._mask_rho

    @mask_rho.deleter
    def _del_mask_rho(self):
        del self._mask_rho
        del self.mask_u
        del self.mask_v
        del self.mask_psi
        del self.mask_vert

    @property
    def mask(self):
        return self.mask_rho

    @mask.setter
    def mask(self, val):
        self.mask_rho = val

    @mask.deleter
    def mask(self):
        del self.mask_rho

    @property
    def mask_is_rho(self):
        return self._mask_is_rho

    @mask_is_rho.setter
    def mask_is_rho(self, mask_is_rho):
        self._mask_is_rho = mask_is_rho
        self.mask_is_u = self._mask_is_rho[:, 1:]*self._mask_is_rho[:, :-1]
        self.mask_is_v = self._mask_is_rho[1:, :]*self._mask_is_rho[:-1, :]
        self.mask_is_psi = \
            self._mask_is_rho[1:, 1:]*self._mask_is_rho[:-1, 1:] * \
            self._mask_is_rho[1:, :-1]*self._mask_is_rho[:-1, :-1]
        self.mask_is_vert = np.ones(self.x_vert.shape, dtype=np.int32)
        for sli in [(slice(-1), slice(1, None)), (slice(1, None), slice(-1)),
                    (slice(-1), slice(-1)), (slice(1, None), slice(1, None))]:
            self.mask_is_vert[sli] *= self._mask_is_rho

    @mask_is_rho.deleter
    def _del_mask_is_rho(self):
        del self._mask_is_rho
        del self.mask_is_u
        del self.mask_is_v
        del self.mask_is_psi
        del self.mask_is_vert

    @property
    def mask_is(self):
        return self.mask_is_rho

    @mask_is.setter
    def mask_is(self, val):
        self.mask_is_rho = val

    @mask_is.deleter
    def mask_is(self):
        del self.mask_is_rho


class CGridGeo(CGrid):
    """
    Curvilinear Arakawa C-grid defined in geographic coordinates

    For a geographic grid, a projection may be specified. The cell widths
    are determined by the great circle distances. Angles, however, are
    defined using the projected coordinates, so a projection that
    conserves angles must be used. This means typically either Mercator
    (projection='merc') or Lambert Conformal Conic (projection='lcc').
    """

    def __init__(self,
                 lon_vert: np.ndarray, lat_vert: np.ndarray,
                 proj: Union[type(None), str, pyproj.Proj],
                 use_gcdist: bool = True, ellipse: str = 'WGS84',
                 lon_rho: _atype = None, lat_rho: _atype = None,
                 lon_psi: _atype = None, lat_psi: _atype = None,
                 lon_u: _atype = None, lat_u: _atype = None,
                 lon_v: _atype = None, lat_v: _atype = None,
                 dx: _atype = None, dy: _atype = None,
                 dndx: _atype = None, dmde: _atype = None,
                 angle_rho: _atype = None, mask_rho: _atype = None):

        self.use_gcdist = use_gcdist
        self.ellipse = ellipse
        if isinstance(proj, pyproj.Proj):
            self.proj = proj
        elif hasattr(proj, 'proj4string'):
            self.proj = pyproj.Proj(proj.proj4string)
        elif proj is None:
            lon0 = (np.sin(lon_vert) + np.cos(lon_vert)*1j).mean()
            lon0 = np.angle(lon0, deg=True)
            lat0 = lat_vert.mean()
            self.proj = pyproj.Proj(proj='aeqd', lon_0=lon0, lat_0=lat0)
        else:
            raise AttributeError('Unknown map projection type.')

        # Pass in lon/lat of vertices
        self._lon_vert = lon_vert
        self._lat_vert = lat_vert

        if self.lon.max() > 180.:
            self._lonr = True
        else:
            self._lonr = False

        # calculate cartesian position
        if lon_rho is None or lat_rho is None or \
           lon_psi is None or lat_psi is None or \
           lon_u is None or lat_u is None or \
           lon_v is None or lat_v is None:

            proj = self.proj
            x_vert, y_vert = proj(lon_vert, lat_vert)
            super(CGridGeo, self).__init__(x_vert, y_vert, mask_rho=mask_rho)
            self.proj = proj

            self.lon_rho, self.lat_rho = \
                self.proj(self.x_rho, self.y_rho, inverse=True)
            self.lon_psi, self.lat_psi = \
                self.proj(self.x_psi, self.y_psi, inverse=True)
            self.lon_u, self.lat_u = \
                self.proj(self.x_u, self.y_u, inverse=True)
            self.lon_v, self.lat_v = \
                self.proj(self.x_v, self.y_v, inverse=True)

            if self._lonr:
                self.lon_rho[self.lon_rho < 0.] += 360.
                self.lon_psi[self.lon_psi < 0.] += 360.
                self.lon_u[self.lon_u < 0.] += 360.
                self.lon_v[self.lon_v < 0.] += 360.

        else:
            self.lon_rho, self.lat_rho = lon_rho, lat_rho
            self.lon_psi, self.lat_psi = lon_psi, lat_psi
            self.lon_u, self.lat_u = lon_u, lat_u
            self.lon_v, self.lat_v = lon_v, lat_v

            self._x_vert, self._y_vert = self.proj(lon_vert, lat_vert)
            self.x_rho, self.y_rho = self.proj(lon_rho, lat_rho)
            self.x_psi, self.y_psi = self.proj(lon_psi, lat_psi)
            self.x_u, self.y_u = self.proj(lon_u, lat_u)
            self.x_v, self.y_v = self.proj(lon_v, lat_v)

        if dx is None or dy is None:
            self._calculate_metrics()
        else:
            self.dx, self.dy = dx, dy
        self.pm, self.pn = 1./self.dx, 1./self.dy
        self.xl = np.maximum(self.dx[0, :].sum(), self.dx[-1, :].sum())
        self.el = np.maximum(self.dy[:, 0].sum(), self.dy[:, -1].sum())

        if dndx is None or dmde is None:
            self._calculate_derivative_metrics()
        else:
            self.dndx, self.dmde = dndx, dmde

        if angle_rho is None:
            self._calculate_angle_rho()
        else:
            self.angle_rho = angle_rho
        self._calculate_angle_xy()

        self.f = 2.0 * 7.29e-5 * np.sin(self.lat_rho * np.pi / 180.0)
        self.spherical = True
        return

    def _calculate_metrics(self):
        """
        Calculate grid and domain lengths pm, pn, xl, el
        """
        if not self.use_gcdist:
            # calculate metrics based on x and y grid
            super(CGridGeo, self)._calculate_metrics()
        else:
            # optionally calculate dx and dy based on great circle
            # distances for more accurate cell sizes.
            geod = pyproj.Geod(ellps=self.ellipse)
            _, _, dx = geod.inv(self.lon[:, 1:],  self.lat[:, 1:],
                                self.lon[:, :-1], self.lat[:, :-1])
            self.dx = 0.5*(dx[1:, :]+dx[:-1, :])
            _, _, dy = geod.inv(self.lon[1:, :],  self.lat[1:, :],
                                self.lon[:-1, :], self.lat[:-1, :])
            self.dy = 0.5*(dy[:, 1:]+dy[:, :-1])
        return

    def _calculate_angle_rho(self):
        """
        Calculate grid angle at rho points
        """
        if not self.use_gcdist:
            # calculate metrics based on x and y grid
            super(CGridGeo, self)._calculate_angle_rho()
        else:
            # optionally calculate angle based on great circle distances
            # for more accurate cell sizes.
            if isinstance(self.lon, np.ma.MaskedArray) or \
               isinstance(self.lat, np.ma.MaskedArray):
                self.angle_rho = np.ma.zeros(self.lon.shape, dtype='d')
            else:
                self.angle_rho = np.zeros(self.lon.shape, dtype='d')

            geod = pyproj.Geod(ellps=self.ellipse)
            angle, _, _ = geod.inv(
                self.lon[:, :-1], self.lat[:, :-1],
                self.lon[:, 1:], self.lat[:, 1:])
            angle = (90.-angle)*np.pi/180.
            angle = np.cos(angle) + np.sin(angle)*1j
            angle = np.angle(0.5*(angle[1:, :] + angle[:-1, :]))
            self.angle_rho = angle
        return

    def _calculate_angle_xy(self):
        self.angle_xy = np.arctan2(
            np.diff(0.5*(self.y_vert[1:, :]+self.y_vert[:-1, :])),
            np.diff(0.5*(self.x_vert[1:, :]+self.x_vert[:-1, :])))
        return

    def add_focus(self, x0: float, y0: float,
                  xfac: _ftype = None, yfac: _ftype = None,
                  rx: _ftype = 0.1, ry: _ftype = 0.1,
                  latlon: bool = True):
        """
        Add local intensification for CGrid. This function will
        regenerate a new set of CGrid.

        x0, y0     - location to focus on, default in lon/lat coordinates.
                     when the kwarg latlon == False, it is in x/y coordinates.
        xfac, yfac - intensification factors in xi/eta direction.
        rx, ry     - normalized range of intensification.
        """
        if latlon:
            x0, y0 = self.proj(x0, y0)
        self._refocus(x0, y0, xfac, yfac, rx, ry)
        lon_vert, lat_vert = self.proj(self.Gridgen.x, self.Gridgen.y,
                                       inverse=True)
        self.__init__(lon_vert, lat_vert, self.proj)
        return

    def mask_land(self, use_iceshelf=False, scale='10m'):
        """
        Mask land/iceshelf points using cartopy coastpolygons.
        This function iterate through the coastpolygons object and call
        mask_polygon() to mask out land/iceshelf points.
        """
        print('Constructing land mask...')
        if not hasattr(self, 'coastpolygons'):
            # Need to fetch coast polygons
            self.get_coastpolygons(scale=scale)
        npoly = len(self.coastpolygons)
        print('Total # of coast polygons: %4d' % npoly)
        for i, pol in enumerate(self.coastpolygons):
            if i != npoly - 1:
                print('processing: %4d/%4d' % (i+1, npoly), end='\r')
            else:
                print('processing: %4d/%4d' % (i+1, npoly))
                print('all coast polygons processed.')
            self.mask_polygon(np.array(pol, np.float32).T)

        if use_iceshelf:
            self.mask_iceshelf(scale=scale)
        return

    def mask_iceshelf(self, scale='10m'):
        print('Constructing iceshelf mask...')
        if not hasattr(self, 'mask_is'):
            self.mask_is = \
                np.zeros(self.mask_rho.shape, dtype=np.int32)
        if not hasattr(self, 'coastpolygons_iceshelf'):
            # Need to fetch iceshelf boundary polygons
            self.get_coastpolygons(use_iceshelf=True, scale=scale)

        npoly = len(self.coastpolygons_iceshelf)
        print('Total # of iceshelf polygons: %4d' % npoly)
        for i, pol in enumerate(self.coastpolygons_iceshelf):
            if i != npoly - 1:
                print('processing: %4d/%4d' % (i+1, npoly), end='\r')
            else:
                print('processing: %4d/%4d' % (i+1, npoly))
                print('all iceshelf polygons processed.')
            self.mask_polygon(np.array(pol, np.float32).T,
                              use_iceshelf=True, mask_value=1)
        return

    def get_coastpolygons(self, use_iceshelf=False, scale='10m'):
        """
        Get coastpolygons from natural earth land product.
        """
        # Get grid central lat/lon
        x0, y0 = self.x.mean(), self.y.mean()
        lon0, lat0 = self.proj(x0, y0, inverse=True)

        # Restrict the center of stereographic projection to the nearest
        # one of 6 points on the sphere (every 90 deg lat/lon).
        lon0 = 90.*(np.around(lon0/90.))
        lat0 = 90.*(np.around(lat0/90.))
        if abs(int(lat0)) == 90:
            lon0 = 0.

        # Check if the projection center is at North Pole
        if lat0 == 90.:
            NPole = True
        else:
            NPole = False

        # Construct Azimuthal projection
        proj_az = pyproj.Proj(proj='aeqd', lat_0=lat0, lon_0=lon0)
        ptrans = pyproj.Transformer.from_proj(
            self.proj, proj_az, always_xy=True)
        ptrans_inv = pyproj.Transformer.from_proj(
            proj_az, self.proj, always_xy=True)
        ptrans_ne = pyproj.Transformer.from_proj(
            pyproj.Proj('epsg:4326'), proj_az, always_xy=True)

        # Transform X/Y from grid projection to Azimuthal projection
        x, y = ptrans.transform(self.x, self.y)

        # Construct grid boundary shapely polygon
        xb = np.concatenate(
            (x[1:, 0], x[-1, 1:], x[-2::-1, -1], x[0, -2::-1]))
        yb = np.concatenate(
            (y[1:, 0], y[-1, 1:], y[-2::-1, -1], y[0, -2::-1]))
        bpoly = sgeometry.Polygon(list(zip(xb, yb)))

        # Get land/iceshelf polygons from natural earth shape file
        if use_iceshelf:
            if scale == '110m':
                scale_iceshelf = '50m'
            else:
                scale_iceshelf = scale
            filename = cio.shapereader.natural_earth(
                resolution=scale_iceshelf, name='antarctic_ice_shelves_polys')
        else:
            filename = cio.shapereader.natural_earth(
                resolution=scale, name='land')
        shp = cio.shapereader.Reader(filename)

        # Convert cartopy shape BasicReader to shapely polygons. 'geom'
        # is an iterable object containing all the coast polygons.
        geom = shp.geometries()

        # Unwrap geom into a list of shapely polygons, and check if each
        # polygon intersects the grid boundary.
        polys = []
        for gi in geom:
            gitrans = sops.transform(ptrans_ne.transform, gi)
            if isinstance(gitrans, sgeometry.MultiPolygon):
                for gitransi in gitrans.geoms:
                    if gitransi.intersects(bpoly):
                        polys.append(gitransi)
            else:
                if gitrans.intersects(bpoly):
                    polys.append(gitrans)

        # Get X/Y coords from the polys and transform to grid projection
        coastpolygons = []
        if NPole:
            # If projection center is at North pole, hack to avoid having
            # Antarctica polygon covering entire map. The idea is to use
            # a consistent 'ocean point', and check if this point is
            # inside the given polygon. If so, then it is the Antarctica
            # polygon and should be ruled out.
            ocean_point = sgeometry.Point(proj_az(0, -45))
            for pi in polys:
                if not ocean_point.within(pi):
                    pi_inv = sops.transform(ptrans_inv.transform, pi)
                    x, y = pi_inv.exterior.coords.xy
                    coastpolygons.append([x, y])
        else:
            for pi in polys:
                pi_inv = sops.transform(ptrans_inv.transform, pi)
                x, y = pi_inv.exterior.coords.xy
                coastpolygons.append([x, y])
        if use_iceshelf:
            self.coastpolygons_iceshelf = coastpolygons
        else:
            self.coastpolygons = coastpolygons
        return

    def mask_polygon_geo(self, lonlat_verts, mask_value=0):
        lon, lat = list(zip(*lonlat_verts))
        x, y = proj(lon, lat, inverse=True)
        self.mask_polygon(list(zip(x, y)), mask_value)
        return

    def longitude_wrap(self):
        lon = self.lon.copy()
        if lon.max() > 180.:
            lon = (lon + 180) % 360 - 180
            self.__init__(lon, self._lat_vert, self.proj,
                          mask_rho=self.mask_rho)
        elif lon.min() < 0.:
            lon = lon % 360
            self.__init__(lon, self._lat_vert, self.proj,
                          mask_rho=self.mask_rho)
        return

    # property decorators
    lon = property(lambda self: self.lon_vert)
    lat = property(lambda self: self.lat_vert)

    @property
    def lon_vert(self):
        return self._lon_vert

    @lon_vert.setter
    def lon_vert(self, lon_vert):
        self.__init__(lon_vert, self._lat_vert, self.proj)

    @property
    def lat_vert(self):
        return self._lat_vert

    @lat_vert.setter
    def lat_vert(self, lat_vert):
        self.__init__(self._lon_vert, lat_vert, self.proj)

    @property
    def x_vert(self):
        return self._x_vert

    @x_vert.setter
    def x_vert(self, x_vert):
        lon_vert, lat_vert = self.proj(x_vert, self._y_vert, inverse=True)
        self.__init__(lon_vert, lat_vert, self.proj)

    @property
    def y_vert(self):
        return self._y_vert

    @y_vert.setter
    def y_vert(self, y_vert):
        lon_vert, lat_vert = self.proj(self._x_vert, y_vert, inverse=True)
        self.__init__(lon_vert, lat_vert, self.proj)


class BoundaryInteractor:
    """
    Interactive grid generation tool.

    bry = BoundaryInteractor(x=[], y=[], beta=None, ax=plt.gca(), proj=proj,
                             **kwargs)

    Inputs:

        x, y   - Iitial boundary polygon points (x and y), counterclockwise,
                 starting in the upper left corner of the boundary.
                 Optionally, x can be a string of filename that contains
                 x/y/beta values.
        beta   - Vortices of x/y points.
        ax     - axes to plot the vertices and grid mesh.
        proj   - pyPROJ projection object. The true x/y coordinates are
                 transformed from original x/y by x, y = proj(x, y)
        kwargs - other keyword arguments to be passed to Gridgen.

    Key-binding commands:

        t : toggle visibility of verticies
        d : delete a vertex
        i : insert a vertex at a point on the polygon line
        u : set upper left corner

        p : set vertex as beta=1 (a Positive turn, marked with green triangle)
        m : set vertex as beta=-1 (a Negative turn, marked with red triangle)
        z : set vertex as beta=0 (no corner, marked with a black dot)

        G : generate grid from the current boundary using gridgen
        T : toggle visability of the current grid
        N : close plot and execute next step

    Attributes:
        bry.x : the X boundary points
        bry.y : the Y boundary points
        bry.verts : the verticies of the grid
        bry.grd : the CGrid/CgridGeo object
    """

    # boolean variables setting visibility of vertices
    _showverts, _showbetas, _showgrid = True, True, True

    # max pixel distance to count as a vertex hit
    _epsilon = 10

    def __init__(self,
                 x: Union[str, List] = [-80, -80, 80, 80],
                 y: List = [60, -60, -60, 60],
                 beta: Union[type(None), List] = None,
                 proj: Union[type(None), pyproj.Proj] = None,
                 ax=None,
                 **kwargs):

        if isinstance(x, str):
            # If pass in a file name, read vertex coordinates from file
            bry_dict = np.load(x)
            x = bry_dict['x']
            y = bry_dict['y']
            beta = bry_dict['beta']
        else:
            # Covert vertex coordinates to list and check coordinate length
            if not isinstance(x, list):
                x = list(x)
            if not isinstance(y, list):
                y = list(y)
            if beta is None:
                if len(x) == 4:
                    beta = [1 for xi in x]
                else:
                    beta = [0 for xi in x]
            else:
                if not isinstance(beta, list):
                    beta = list(beta)

        assert len(x) >= 4, 'Boundary must have at least four points.'
        assert len(x) == len(y), 'Vortex X/Y coords must have the same length.'
        assert len(x) == len(beta), 'beta must have same length as X/Y'

        # Pass in beta and projection
        self.beta, self.proj = beta, proj

        # Pass in ploting axes
        if ax is None:
            ax = plt.gca()
        self._ax, self._canvas = ax, ax.figure.canvas
        self._ax.set_title('Boundary Interactor')

        # Set default gridgen option, and copy over specified options.
        self._gridgen_options = {'ul_idx': 0, 'shp': (32, 32)}
        for key in kwargs.keys():
            self._gridgen_options[key] = kwargs[key]

        # Set the default line and polygon objects
        self._line = Line2D(x, y, ls='-', color='k', lw=1, animated=True)
        self._ax.add_line(self._line)
        self._poly = Polygon(self.verts, alpha=0.1, fc='k', animated=True)
        self._ax.add_patch(self._poly)

        # Link in the lines that will show the beta values.
        # _pline for positive turns, _mline for negative (minus) turns.
        # otherwize _zline (zero) for straight sections.
        line_options = dict(ms=8, lw=0, animated=True)
        self._pline = Line2D([], [], marker='^', mfc='g', **line_options)
        self._mline = Line2D([], [], marker='v', mfc='r', **line_options)
        self._zline = Line2D([], [], marker='o', mfc='k', **line_options)
        self._sline = Line2D([], [], marker='s', mfc='k', **line_options)

        self._update_beta_lines()
        self._ax.add_line(self._pline)
        self._ax.add_line(self._mline)
        self._ax.add_line(self._zline)
        self._ax.add_line(self._sline)

        # get the canvas and connect the callback events.
        # The active vert
        self.cid = self._poly.add_callback(self._poly_changed)
        self._ind = None
        self._canvas.mpl_connect('draw_event',
                                 self._on_draw)
        self._canvas.mpl_connect('button_press_event',
                                 self._on_button_press)
        self._canvas.mpl_connect('button_release_event',
                                 self._on_button_release)
        self._canvas.mpl_connect('motion_notify_event',
                                 self._on_motion_notify)
        self._canvas.mpl_connect('key_press_event',
                                 self._on_key_press)

        # Print the command line instructions on screen
        print('Boundary Interactor Commands:')
        print('')
        print('   t : toggle visibility of verticies')
        print('   d : delete a vertex')
        print('   i : insert a vertex at a point on the polygon line')
        print('   u : set upper left corner')
        print('')
        print('   p : set vertex as beta=1 ' +
              '(a Positive turn, marked with green triangle)')
        print('   m : set vertex as beta=-1 ' +
              '(a Negative turn, marked with red triangle)')
        print('   z : set vertex as beta=0 ' +
              '(no corner, marked with a black dot)')
        print('')
        print('   G : generate grid from the current boundary using gridgen')
        print('   T : toggle visability of the current grid')
        print('   N : close plot and execute next step')
        print('')

        plt.show()

    def _on_draw(self, event):
        """
        Update boundary vertices when firstly drawn.
        """
        self._background = self._canvas.copy_from_bbox(self._ax.bbox)
        self._ax.draw_artist(self._poly)
        self._ax.draw_artist(self._line)
        self._ax.draw_artist(self._pline)
        self._ax.draw_artist(self._mline)
        self._ax.draw_artist(self._zline)
        self._ax.draw_artist(self._sline)
        return

    def _poly_changed(self, poly):
        """
        This method is called whenever the pathpatch object is called.
        """
        # only copy the artist props to the line (except visibility)
        vis = self._line.get_visible()
        Artist.update_from(self._line, poly)
        self._line.set_visible(vis)  # don't use the poly visibility state
        return

    def _on_button_press(self, event):
        """
        Get vertex index at click point.
        """
        if not self._showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        # get vertex index
        self._ind = self._get_ind_under_point(event)
        return

    def _on_button_release(self, event):
        """
        Release selected vertex.
        """
        if not self._showverts:
            return
        if event.button != 1:
            return
        self._ind = None
        return

    def _on_motion_notify(self, event):
        """
        Drag vertex over map.
        """
        if not self._showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        # transform the polygon as mouse moves
        x, y = event.xdata, event.ydata
        self._poly.xy[self._ind] = x, y
        if self._ind == 0:
            self._poly.xy[-1] = x, y
        elif self._ind == len(self._poly.xy) - 1:
            self._poly.xy[0] = x, y

        self._line.set_data(zip(*self._poly.xy[:-1]))
        self._update_beta_lines()

        self._canvas.restore_region(self._background)
        self._ax.draw_artist(self._poly)
        self._ax.draw_artist(self._line)
        self._ax.draw_artist(self._pline)
        self._ax.draw_artist(self._mline)
        self._ax.draw_artist(self._zline)
        self._ax.draw_artist(self._sline)
        self._canvas.blit(self._ax.bbox)
        return

    def _on_key_press(self, event):
        """
        Key press events.
        """
        ind = self._get_ind_under_point(event)

        if event.key == 't':
            self._showbetas = not self._showbetas
            if self._showbetas:
                print('Showing verticies')
            else:
                print('Hiding verticies')
            self._line.set_visible(self._showbetas)
            self._pline.set_visible(self._showbetas)
            self._mline.set_visible(self._showbetas)
            self._zline.set_visible(self._showbetas)
            self._sline.set_visible(self._showbetas)
        elif event.key == 'd':
            if ind is not None:
                print('Deleting vertex')
                self.beta.pop(ind)
                self._poly.xy = np.delete(self._poly.xy, ind, axis=0)
                self._line.set_data(zip(*self._poly.xy[:-1]))
        elif event.key == 'p':
            if ind is not None:
                print('Setting vertex as beta=1')
                self.beta[ind] = 1.0
        elif event.key == 'm':
            if ind is not None:
                print('Setting vertex as beta=-1')
                self.beta[ind] = -1.0
        elif event.key == 'z':
            if ind is not None:
                print('Setting vertex as beta=0')
                self.beta[ind] = 0.0
        elif event.key == 'u':
            if ind is not None:
                print('Setting upper left corner')
                self._gridgen_options['ul_idx'] = ind
        elif event.key == 'i':
            print('Inserting new vertex')
            xys = self._poly.get_transform().transform(self._poly.xy)
            p = event.x, event.y
            for i in range(len(xys)-1):
                s0, s1 = xys[i], xys[i+1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self._epsilon:
                    self._poly.xy = np.insert(
                        self._poly.xy, i+1, [event.xdata, event.ydata], axis=0)
                    self._line.set_data(zip(*self._poly.xy[:-1]))
                    self.beta.insert(i+1, 0)
                    break
        elif event.key == 'G' or event.key == '1':
            print('Constructing C-Grid...')
            gridgen_options = self._gridgen_options.copy()
            shp = gridgen_options.pop('shp')
            self.grd = gridgen(self.x, self.y, self.beta, shp,
                               proj=self.proj, **gridgen_options)

            self._remove_grid()
            self._showgrid = True
            if max(shp) <= 100:
                lw = 0.7
            elif max(shp) <= 200:
                lw = 0.5
            elif max(shp) <= 500:
                lw = 0.3
            else:
                lw = 0.1
            gridlineprops = dict(linestyle='-', color='k', lw=lw)
            self._gridlines = []
            for line in self._ax._get_lines(*(self.grd.x, self.grd.y),
                                            **gridlineprops):
                self._ax.add_line(line)
                self._gridlines.append(line)
            for line in self._ax._get_lines(*(self.grd.x.T, self.grd.y.T),
                                            **gridlineprops):
                self._ax.add_line(line)
                self._gridlines.append(line)
        elif event.key == 'T' or event.key == '2':
            self._showgrid = not self._showgrid
            if self._showgrid:
                print('Showing grid lines')
            else:
                print('Hiding grid lines')
            if hasattr(self, '_gridlines'):
                for line in self._gridlines:
                    line.set_visible(self._showgrid)
        elif event.key == 'N':
            plt.close()
            return

        self._update_beta_lines()
        if self._line.stale or \
           self._pline.stale or self._mline.stale or \
           self._zline.stale or self._sline.stale:
            self._canvas.draw_idle()

        return

    def _update_beta_lines(self):
        """
        Update m/p-line by finding the points where self.beta== -/+ 1
        """
        x, y = list(zip(*self._poly.xy))

        # the first and last point are repeated
        num_points = len(x) - 1

        # update p, m, z vertices
        xp = [x[n] for n in range(num_points) if self.beta[n] == 1]
        yp = [y[n] for n in range(num_points) if self.beta[n] == 1]
        self._pline.set_data(xp, yp)

        xm = [x[n] for n in range(num_points) if self.beta[n] == -1]
        ym = [y[n] for n in range(num_points) if self.beta[n] == -1]
        self._mline.set_data(xm, ym)

        xz = [x[n] for n in range(num_points) if self.beta[n] == 0]
        yz = [y[n] for n in range(num_points) if self.beta[n] == 0]
        self._zline.set_data(xz, yz)

        # update upper-left conor vertex
        if len(x)-1 < self._gridgen_options['ul_idx']:
            self._gridgen_options['ul_idx'] = len(x)-1
        xs = x[self._gridgen_options['ul_idx']]
        ys = y[self._gridgen_options['ul_idx']]
        self._sline.set_data(xs, ys)
        return

    def _get_ind_under_point(self, event):
        """
        get the index of the vertex under point if within epsilon tolerance
        """
        xy = self._poly.xy
        xy = self._poly.get_transform().transform(xy)
        x, y = xy[:, 0], xy[:, 1]

        # calculate distance
        d = np.hypot(x - event.x, y - event.y)
        ind = d.argmin()
        if d[ind] >= self._epsilon:
            ind = None

        return ind

    def _remove_grid(self):
        """
        Remove previously generated grid from the BoundaryInteractor figure
        """
        if hasattr(self, '_gridlines'):
            for line in self._gridlines:
                self._ax.lines.remove(line)
            delattr(self, '_gridlines')

    @property
    def x(self):
        return self._line.get_xdata()

    @property
    def y(self):
        return self._line.get_ydata()

    @property
    def verts(self):
        return list(zip(self.x, self.y))


def dist(x: _fatype, y: _fatype) -> _fatype:
    """
    Return the distance between two points.
    """
    d = x - y
    return np.sqrt(np.dot(d, d))


def dist_point_to_segment(p: _fatype, s0: _fatype, s1: _fatype) -> _fatype:
    """
    Get the distance of a point to a segment.
      *p*, *s0*, *s1* are *xy* sequences
    This algorithm from http://geomalgorithms.com/a02-_lines.html
    """
    v = s1 - s0
    w = p - s0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return dist(p, s0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return dist(p, s1)
    b = c1 / c2
    pb = s0 + b * v
    return dist(p, pb)


class Gridgen:
    """
    Gridgen interface.
    """
    def __init__(self, xbry, ybry, beta, shape, ul_idx=0,
                 focus=None, proj=None, nnodes=14, precision=1.0e-12, nppe=3,
                 newton=True, thin=True, checksimplepoly=True, verbose=False):

        foundit = False
        conda_lib_path = os.getenv('CONDA_PREFIX') + '/lib'
        if len(glob.glob(conda_lib_path + '/libgridgen*.so')) > 0:
            self._libgridgen = \
                np.ctypeslib.load_library('libgridgen', conda_lib_path)
            foundit = True
        else:
            ld_lib_path_list = os.getenv('LD_LIBRARY_PATH')
            ld_lib_path_list = ld_lib_path_list.split(':')
            for ld_lib_path in ld_lib_path_list:
                if len(glob.glob(ld_lib_path + '/libgridgen*.so')) > 0:
                    self._libgridgen = \
                        np.ctypeslib.load_library('libgridgen', ld_lib_path)
                    foundit = True
                    break
        if not foundit:
            self._libgridgen = \
                np.ctypeslib.load_library('libgridgen', pyroms.__path__[0])
            foundit = True

        class GRIDSTATS(ctypes.Structure):
            _fields_ = [
                ("mdo", ctypes.c_double),
                ("imdo", ctypes.c_int),
                ("jmdo", ctypes.c_int),
                ("ado", ctypes.c_double),
                ("mar", ctypes.c_double),
                ("aar", ctypes.c_double)]

        class GRIDNODES(ctypes.Structure):
            _fields_ = [
                ("nx", ctypes.c_int),
                ("ny", ctypes.c_int),
                ("gx", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ("gy", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ("type", ctypes.c_int),
                ("validated", ctypes.c_int),
                ("stats", ctypes.POINTER(GRIDSTATS)),
                ("nextpoint", ctypes.c_int)]

        class EXTENT(ctypes.Structure):
            _fields_ = [
                ("xmin", ctypes.c_double),
                ("xmax", ctypes.c_double),
                ("ymin", ctypes.c_double),
                ("ymax", ctypes.c_double)]

        class POLY(ctypes.Structure):
            _fields_ = [
                ("n", ctypes.c_int),
                ("nallocated", ctypes.c_int),
                ("e", EXTENT),
                ("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double))]

        # SUBGRID
        # A forward declaration of this structure is used
        # (1) Defined it first with pass
        # (2) Define the fields next

        # SUBGRID (1)
        class SUBGRID(ctypes.Structure):
            pass

        class GRIDMAP(ctypes.Structure):
            _fields_ = [
                ("bound", ctypes.POINTER(POLY)),
                ("trunk", ctypes.POINTER(SUBGRID)),
                ("nleaves", ctypes.c_int),
                ("nce1", ctypes.c_int),
                ("nce2", ctypes.c_int),
                ("gx", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ("gy", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ("sign", ctypes.c_int)]

        # SUBGRID (2)
        SUBGRID._fields_ = [
            ("gmap", ctypes.POINTER(GRIDMAP)),
            ("bound", ctypes.POINTER(POLY)),
            ("mini", ctypes.c_int),
            ("maxi", ctypes.c_int),
            ("minj", ctypes.c_int),
            ("maxj", ctypes.c_int),
            ("half1", ctypes.POINTER(SUBGRID)),
            ("half2", ctypes.POINTER(SUBGRID))]

        self._libgridgen.gridgen_generategrid2.restype = \
            ctypes.POINTER(GRIDNODES)
        self._libgridgen.gridnodes_getx.restype = \
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
        self._libgridgen.gridnodes_gety.restype = \
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
        self._libgridgen.gridnodes_getnce1.restype = ctypes.c_int
        self._libgridgen.gridnodes_getnce2.restype = ctypes.c_int
        self._libgridgen.gridmap_build.restype = ctypes.POINTER(GRIDMAP)

        self.xbry = np.asarray(xbry, dtype='d')
        self.ybry = np.asarray(ybry, dtype='d')
        self.beta = np.asarray(beta, dtype='d')
        assert self.beta.sum() == 4.0, 'sum of beta must be 4.0'
        self.shape = shape
        self.ny = shape[0]
        self.nx = shape[1]
        self.ul_idx = ul_idx
        self.focus = focus
        self.nnodes = nnodes
        self.precision = precision
        self.nppe = nppe
        self.newton = newton
        self.thin = thin
        self.checksimplepoly = checksimplepoly
        self.verbose = verbose

        if proj is not None:
            self.xbry, self.ybry = proj(self.xbry, self.ybry)

        self._gn = None
        self.__call__()
        return

    def __call__(self):

        nbry = len(self.xbry)

        nsigmas = ctypes.c_int(0)
        sigmas = ctypes.c_void_p(0)
        nrect = ctypes.c_int(0)
        xrect = ctypes.c_void_p(0)
        yrect = ctypes.c_void_p(0)

        if self.focus is None:
            ngrid = ctypes.c_int(0)
            xgrid = ctypes.POINTER(ctypes.c_double)()
            ygrid = ctypes.POINTER(ctypes.c_double)()
        else:
            y, x = np.mgrid[0:1:self.ny*1j, 0:1:self.nx*1j]
            xgrid, ygrid = self.focus(x, y)
            ngrid = ctypes.c_int(xgrid.size)
            xgrid = (ctypes.c_double * xgrid.size)(*xgrid.flatten())
            ygrid = (ctypes.c_double * ygrid.size)(*ygrid.flatten())

        self._gn = self._libgridgen.gridgen_generategrid2(
            ctypes.c_int(nbry),
            (ctypes.c_double * nbry)(*self.xbry),
            (ctypes.c_double * nbry)(*self.ybry),
            (ctypes.c_double * nbry)(*self.beta),
            ctypes.c_int(self.ul_idx),
            ctypes.c_int(self.nx),
            ctypes.c_int(self.ny),
            ngrid,
            xgrid,
            ygrid,
            ctypes.c_int(self.nnodes),
            ctypes.c_int(self.newton),
            ctypes.c_double(self.precision),
            ctypes.c_int(self.checksimplepoly),
            ctypes.c_int(self.thin),
            ctypes.c_int(self.nppe),
            ctypes.c_int(self.verbose),
            ctypes.byref(nsigmas),
            ctypes.byref(sigmas),
            ctypes.byref(nrect),
            ctypes.byref(xrect),
            ctypes.byref(yrect))

        x = self._libgridgen.gridnodes_getx(self._gn)
        x = np.asarray([x[0][i] for i in range(self.ny*self.nx)])
        x.shape = (self.ny, self.nx)

        y = self._libgridgen.gridnodes_gety(self._gn)
        y = np.asarray([y[0][i] for i in range(self.ny*self.nx)])
        y.shape = (self.ny, self.nx)

        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            x = np.ma.masked_where(np.isnan(x), x)
            y = np.ma.masked_where(np.isnan(y), y)

        self.x, self.y = x, y
        return

    def __del__(self):
        """
        delete gridnode object upon deletion
        """
        self._libgridgen.gridnodes_destroy(self._gn)
        return


def gridgen(xbry: List, ybry: List, beta: List, shape: Tuple,
            ul_idx=0, focus=None, proj=None, nnodes=14,
            precision=1.0e-12, nppe=3, newton=True, thin=True,
            checksimplepoly=True, verbose=False):
    """
    External wrapping function to call Gridgen grid builder.
    xbry, ybry - nodes coordinates of grid boundary
    beta       - vertex type
    shape      - tuple of grid shape (eta, xi)
    """
    # Prepare the Gridgen object.
    gn = Gridgen(xbry, ybry, beta, shape,
                 ul_idx=ul_idx, focus=focus, proj=None, nnodes=nnodes,
                 precision=precision, nppe=nppe, newton=newton, thin=thin,
                 checksimplepoly=checksimplepoly, verbose=verbose)
    # Generate the C-Grid.
    if proj is not None:
        lon_vert, lat_vert = proj(gn.x, gn.y, inverse=True)
        grd = CGridGeo(lon_vert, lat_vert, proj)
    else:
        grd = CGrid(gn.x, gn.y)
    # Attach the Gridgen object to grid.
    grd.Gridgen = gn
    print('Grid construction complete.')
    return grd


class EditMaskIJ:
    """
    Interactive mask editor

    EditMaskIJ(grd, coast=None, file_out='mask_change.txt')

    Edit grd mask. Mask/Unsmask cell by a simple click on the cell.
    Mask modification are store in mask_change.txt for further use.

    Key-binding commands:
        e : toggle between Editing/Viewing mode
        N : end mask editing mode
    """

    def __init__(self, grd, coast=None, iceshelf=False,
                 file_out='mask_change.txt'):

        if isinstance(grd, pyroms.grid.ROMSGrid):
            self.hgrid = grd.hgrid
        elif isinstance(grd, CGrid):
            self.hgrid = grd

        # Pass input parameters to self
        self._file_out = file_out
        self._iceshelf = iceshelf

        self.mask = self.hgrid.mask_rho

        # Land/Ocean color for mask editor GUI
        self.land_color = (0.6, 1.0, 0.6)
        self.sea_color = (0.6, 0.6, 1.0)
        self.cm = plt.matplotlib.colors.ListedColormap(
            [self.land_color, self.sea_color], name='land/sea')

        # The coast polygons are fetched from Cartopy's default, Natural
        # Earth topgraphy dataset
        if coast is None:
            if not hasattr(self, 'coastpolygons'):
                self.hgrid.get_coastpolygons()
            coast = pyroms.utility.get_coast_from_coastpolygons(
                self.hgrid.coastpolygons)
        self.coast = coast
        self.ijcoast = pyroms.utility.get_ijcoast(coast, grd)

        if self._iceshelf:
            # If use iceshelf, Pass in iceshelf masks and redefine mask
            # values
            self.mask_rho = self.hgrid.mask_rho
            self.mask_is = self.hgrid.mask_is
            self.mask = np.maximum.reduce((2*(1-self.mask_rho),
                                           self.mask_is))
            self.iceshelf_color = (1.0, 0.6, 0.6)
            self.cm = plt.matplotlib.colors.ListedColormap(
                [self.sea_color, self.iceshelf_color, self.land_color],
                name='land/sea/iceshelf')

        return

    def __call__(self, ax=None, **kwargs):

        # Grid vertex and center coordinate
        x = np.arange(self.mask.shape[1]+1)
        y = np.arange(self.mask.shape[0]+1)
        self.xv, self.yv = np.meshgrid(x, y)

        if ax is None:
            ax = plt.gca()
        self._ax, self._canvas = ax, ax.figure.canvas
        shp = self.xv.shape

        self._pc = ax.imshow(
            self.mask, cmap=self.cm, vmin=0.0, vmax=self.cm.N-1,
            extent=(0, shp[1]-1, 0, shp[0]-1),
            origin='lower', interpolation='nearest',
            **kwargs)
        ax.plot(self.xv, self.yv, color='k', ls='-', lw=.1)
        ax.plot(self.xv.T, self.yv.T, color='k', ls='-', lw=.1)
        ax.plot(self.ijcoast[:, 0], self.ijcoast[:, 1],
                color='k', ls='-', lw=0.5)

        self._ax_setting()

        plt.show()
        return

    def _ax_setting(self):
        self._clicking = False
        self._ax.set_title(
            'Editing %s -- click "e" to toggle' % self._clicking)
        self._canvas.mpl_connect('button_press_event', self._on_button_press)
        self._canvas.mpl_connect('key_press_event', self._on_key_press)

        print('Mask editing commands:')
        print('    left mouse click: Change Land Mask')
        if self._iceshelf:
            print('    right mouse click: Change Iceshelf Mask')
        print('    e: Toggle between Editing/Viewing mode')
        print('    N: End mask editing mode')
        return

    def _on_key_press(self, event):
        if event.key == 'e':
            self._clicking = not self._clicking
            self._ax.set_title(
                'Editing %s -- click "e" to toggle' % self._clicking)
            self._canvas.draw()
            if self._clicking:
                print('Change Mask:     i,   j, maskVal')
            if not self._clicking:
                print('End Change Mask')
        elif event.key == 'N':
            print('Ending mask editing...')
            plt.close()
            return
        return

    def _on_button_press(self, event):
        if event.inaxes and self._clicking:
            j, i = int(event.xdata), int(event.ydata)
        else:
            return

        pc_update = self._update_mask(j, i, event.button)
        if pc_update:
            self._pc.set_array(self.mask)
            self._pc.changed()
            self._canvas.draw()
        return

    def _update_mask(self, j, i, button):
        pc_update = False
        if self._iceshelf:
            if button == 1:
                self.mask_rho[i, j] = 1 - self.mask_rho[i, j]
                self.mask[i, j] = max(2*(1 - self.mask_rho[i, j]),
                                      self.mask_is[i, j])
                # write to output file
                f = open(self._file_out, 'a')
                s = r'%d %d %f' % (i, j, self.mask_rho[i, j])
                f.write(s + '\n')
                f.close()
                s_fmt = r'%4d %4d %6d' % (i, j, int(self.mask_rho[i, j]))
                print('Change Mask: ', s_fmt)
                pc_update = True
            elif button == 3:
                self.mask_is[i, j] = 1 - self.mask_is[i, j]
                self.mask[i, j] = max(2*(1 - self.mask_rho[i, j]),
                                      self.mask_is[i, j])
                # write to output file
                f = open(self._file_out, 'a')
                s = r'%d %d %f' % (i, j, self.mask_is[i, j])
                f.write(s + '\n')
                f.close()
                s_fmt = r'%4d %4d %6d' % (i, j,
                                          int(self.mask_is[i, j]))
                print('Change Iceshelf Mask: ', s_fmt)
                pc_update = True
        else:
            if button == 1:
                self.mask[i, j] = 1 - self.mask[i, j]

                # write to output file
                f = open(self._file_out, 'a')
                s = r'%d %d %f' % (i, j, self.mask[i, j])
                f.write(s + '\n')
                f.close()
                s_fmt = r'%4d %4d %6d' % (i, j, int(self.mask[i, j]))
                print('Change Mask: ', s_fmt)
                pc_update = True

        return pc_update


class EditMask(EditMaskIJ):
    """
    Interactive mask editor

    EditMask(grd, proj=None, coast=None, file_out='mask_change.txt')

    Edit grd mask. Mask/Unsmask cell by a simple click on the cell.
    Mask modification are store in mask_change.txt for further use.

    Key commands:
        e : toggle between Editing/Viewing mode
        N : end mask editing mode
    """

    def __call__(self, geo=False, ax=None, **kwargs):

        # If use geological coordinates, attempt to fetch projection
        # info from ROMS hgrid.
        if geo:
            self.proj = self.hgrid.proj
        else:
            self.proj = None

        if self.proj is None:
            self.xv = self.hgrid.x_vert
            self.yv = self.hgrid.y_vert
            if ax is None:
                ax = plt.gca()
            ax.plot(self.coast[:, 0], self.coast[:, 1],
                    color='k', ls='-', lw=0.5)
        else:
            lon, lat = self.proj(self.hgrid.x_rho.mean(), self.y_rho.mean(),
                                 inverse=True)
            # Only support Stereographic projection for now
            self._proj_az = pyproj.Proj(
                proj='stere', lon_0=lon, lat_0=lat)
            self._mproj = ccrs.Stereographic(
                central_latitude=lat, central_longitude=lon,
                false_easting=0.0, false_northing=0.0)
            self.xv, self.yv = \
                self._proj_az(self.hgrid.lon_vert, self.hgrid.lat_vert)

            _, ax = plt.subplots(subplot_kw={'projection': self._mproj})
            ax.coastlines(resolution='10m')

        self._ax, self._canvas = ax, ax.figure.canvas
        self._pc = self._ax.pcolormesh(
            self.xv, self.yv, self.mask, cmap=self.cm,
            vmin=0, vmax=self.cm.N-1, edgecolor='k', linewidth=0.1,
            **kwargs)

        self._xc = 0.25*(self.xv[1:, 1:]+self.xv[1:, :-1] +
                         self.xv[:-1, 1:]+self.xv[:-1, :-1])
        self._yc = 0.25*(self.yv[1:, 1:]+self.yv[1:, :-1] +
                         self.yv[:-1, 1:]+self.yv[:-1, :-1])

        self._ax_setting()

        plt.show()
        return

    def _on_button_press(self, event):
        if event.inaxes and self._clicking:
            d = np.hypot(self._xc - event.xdata, self._yc - event.ydata)
            idx = np.argmin(d)
            i, j = np.unravel_index(idx, d.shape)
        else:
            return

        pc_update = self._update_mask(j, i, event.button)
        if pc_update:
            self._pc.set_array(self.mask.ravel())
            self._pc.changed()
            self._canvas.draw()
        return


class Focus():
    """
    Return a container for a sequence of Focus objects

    foc = Focus(xo=None, yo=None, xfactor=None, yfactor=None, rx=0.1, ry=0.1)

    The sequence is populated by using the 'add_focus_x' and 'add_focus_y'
    methods. These methods define a point ('xo' or 'yo'), around witch to
    focus, a focusing factor of 'focus', and x and y extent of focusing
    given by Rx or Ry. The region of focusing will be approximately
    Gausian, and the resolution will be increased by approximately the
    value of factor.

    Methods
    -------
    foc.add_focus(xo, yo, xfactor=None, yfactor=None, rx=0.1, ry=0.1)
    foc.add_focus_x(xo, factor=2.0, Rx=0.1)
    foc.add_focus_y(yo, factor=2.0, Ry=0.1)

    Calls to the object return transformed coordinates:
        xf, yf = foc(x, y)
    where x and y must be within [0, 1], and are typically a uniform,
    normalized grid. The focused grid will be the result of applying each
    of the focus elements in the sequence they are added to the series.


    EXAMPLES
    --------

    >>> foc = pyroms.grid.Focus()
    >>> foc.add_focus_x(0.2, factor=3.0, Rx=0.2)
    >>> foc.add_focus_y(0.6, factor=5.0, Ry=0.35)

    >>> x, y = np.mgrid[0:1:3j,0:1:3j]
    >>> xf, yf = foc(x, y)

    >>> print(xf)
    [[ 0.          0.          0.        ]
     [ 0.36594617  0.36594617  0.36594617]
     [ 1.          1.          1.        ]]
    >>> print(yf)
    [[ 0.          0.62479833  1.        ]
     [ 0.          0.62479833  1.        ]
     [ 0.          0.62479833  1.        ]]
    """

    def __init__(self,
                 xo=None, yo=None,
                 xfactor=None, yfactor=None,
                 rx=0.1, ry=0.1):
        self._focuspoints = []
        if (xo is not None) and (xfactor is not None):
            self.add_focus_x(xo, factor=xfactor, Rx=rx)
        if (yo is not None) and (yfactor is not None):
            self.add_focus_y(yo, factor=yfactor, Ry=ry)

    def add_focus(self, xo, yo, xfactor=None, yfactor=None, rx=0.1, ry=0.1):
        """
        Add focus method in both x and y direction.

        xo, yo - fractional coordinate (0<=xo, yo<=1) in eta, xi direction.
        xfactor, yfactor - focus intensification factor (>=1).
        rx, ry - fractional range of intensification.
        """
        if xfactor is not None:
            self.add_focus_x(xo, factor=xfactor, Rx=rx)
        if yfactor is not None:
            self.add_focus_y(yo, factor=yfactor, Ry=ry)

    def add_focus_x(self, xo, factor=2.0, Rx=0.1):
        """
        Add focus method in x direction.
        """
        self._focuspoints.append(self._Focus_x(xo, factor, Rx))

    def add_focus_y(self, yo, factor=2.0, Ry=0.1):
        """
        Add focus method in y direction.
        """
        self._focuspoints.append(self._Focus_y(yo, factor, Ry))

    def __call__(self, x, y):
        """
        This method is used by Gridgen.
        """
        for focuspoint in self._focuspoints:
            x, y = focuspoint(x, y)
        return x, y

    class _Focus_x():
        """
        Return a transformed, uniform grid, focused in the x-direction

        This class may be called with a uniform grid, with limits from [0, 1],
        to create a focused grid in the x-directions centered about xo.
        The output grid is also uniform from [0, 1] in both x and y.

        Parameters
        ----------
        xo : float
            Location about which to focus grid
        factor : float
            amount to focus grid. Creates cell sizes that are factor
            smaller in the focused region.
        Rx : float
            Lateral extent of focused region, similar to a lateral spatial
            scale for the focusing region.

        Returns
        -------
        foc : class
            The class may be called with arguments of a grid. The returned
            transformed grid (x, y) will be focused as per the parameters
            above.
        """

        def __init__(self, xo, factor=2.0, Rx=0.1):
            self.xo = xo
            self.factor = factor
            self.Rx = Rx

        def __call__(self, x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            assert not np.any(x > 1.0) or not np.any(x < 0.0)  \
                or not np.any(y > 1.0) or not np.any(x < 0.0), \
                'x and y must both be within the range [0, 1].'

            alpha = 1.0 - 1.0/self.factor

            def xf(x):
                return x - 0.5*(np.sqrt(np.pi)*self.Rx*alpha *
                                _approximate_erf((x-self.xo)/self.Rx))

            xf0 = xf(0.0)
            xf1 = xf(1.0)

            return (xf(x)-xf0)/(xf1-xf0), y

    class _Focus_y():
        """
        Return a transformed, uniform grid, focused in the y-direction

        This class may be called with a uniform grid, with limits from [0, 1],
        to create a focused grid in the y-directions centered about yo.
        The output grid is also uniform from [0, 1] in both x and y.

        Parameters
        ----------
        yo : float
            Location about which to focus grid
        factor : float
            amount to focus grid. Creates cell sizes that are factor
            smaller in the focused region.
        Ry : float
            Lateral extent of focused region, similar to a lateral
            spatial scale for the focusing region.

        Returns
        -------
        foc : class
            The class may be called with arguments of a grid. The returned
            transformed grid (x, y) will be focused as per the parameters
            above.
        """

        def __init__(self, yo, factor=2.0, Ry=0.1):
            self.yo = yo
            self.factor = factor
            self.Ry = Ry

        def __call__(self, x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            assert not np.any(x > 1.0) or not np.any(x < 0.0)  \
                or not np.any(y > 1.0) or not np.any(x < 0.0), \
                'x and y must both be within the range [0, 1].'

            alpha = 1.0 - 1.0/self.factor

            def yf(y):
                return y - 0.5*(np.sqrt(np.pi)*self.Ry*alpha *
                                _approximate_erf((y-self.yo)/self.Ry))

            yf0 = yf(0.0)
            yf1 = yf(1.0)

            return x, (yf(y)-yf0)/(yf1-yf0)


def _approximate_erf(x):
    '''
    Return approximate solution to error function
    see http://en.wikipedia.org/wiki/Error_function
    '''
    a = -(8*(np.pi-3.0)/(3.0*np.pi*(np.pi-4.0)))
    return np.sign(x) * \
        np.sqrt(1.0 - np.exp(-x**2*(4.0/np.pi+a*x*x)/(1.0+a*x*x)))


class GetPositionFromMap(EditMask):
    """
    Get cell index position and coordinate Interactively

    GetPositionFromMap(grd, proj)

    Get index i, j as well as x, y, lon, lat coordinates for one cell
    simply by clicking on the cell.

    Key commands:
        i : toggle between Interactive/Viewing mode
    """

    _epsilon = 20

    def __call__(self, ax=None):

        self.xi, self.eta, self.x, self.y, self.angle = [], [], [], [], []
        if self.hgrid.spherical:
            self.proj = self.hgrid.proj
            self.lon, self.lat = [], []
        else:
            self.proj = None

        if not self.hgrid.spherical:
            self.xv = self.hgrid.x_vert
            self.yv = self.hgrid.y_vert
            if ax is None:
                ax = plt.gca()
            ax.plot(self.coast[:, 0], self.coast[:, 1],
                    color='k', ls='-', lw=0.5)
        else:
            lon, lat = self.proj(
                self.hgrid.x_rho.mean(), self.hgrid.y_rho.mean(), inverse=True)
            # Only support Stereographic projection for now
            self._proj_az = pyproj.Proj(
                proj='stere', lon_0=lon, lat_0=lat)
            self._mproj = ccrs.Stereographic(
                central_latitude=lat, central_longitude=lon,
                false_easting=0.0, false_northing=0.0)
            self.xv, self.yv = \
                self._proj_az(self.hgrid.lon_vert, self.hgrid.lat_vert)

            _, ax = plt.subplots(subplot_kw={'projection': self._mproj})
            ax.coastlines(resolution='10m')
            ax.gridlines(xlocs=range(-180, 180, 10), ylocs=range(-90, 90, 5),
                         linewidth=0.5, draw_labels=True)

        self._ax, self._canvas = ax, ax.figure.canvas
        self._pc = self._ax.pcolormesh(
            self.xv, self.yv, self.mask, cmap=self.cm,
            vmin=0, vmax=self.cm.N-1)
        shp = self.xv.shape
        if max(shp) <= 100:
            lw = 0.7
        elif max(shp) <= 200:
            lw = 0.5
        elif max(shp) <= 500:
            lw = 0.3
        else:
            lw = 0.1
        self._ax.plot(self.xv, self.yv, '-k', lw=lw)
        self._ax.plot(self.xv.T, self.yv.T, '-k', lw=lw)

        self._line = Line2D([], [], marker='o', mfc='k', ms=8,
                            ls='-', color='k', alpha=0.5, lw=1,
                            animated=True)
        self._ax.add_line(self._line)

        self._xc = 0.25*(self.xv[1:, 1:]+self.xv[1:, :-1] +
                         self.xv[:-1, 1:]+self.xv[:-1, :-1])
        self._yc = 0.25*(self.yv[1:, 1:]+self.yv[1:, :-1] +
                         self.yv[:-1, 1:]+self.yv[:-1, :-1])

        self._ind = None
        self._clicking = False
        ax.set_title('Editing %s -- click "e" to toggle' % self._clicking)
        print('    e: toggle between Editing/Viewing mode')
        print('    N: end editing')

        self._canvas.mpl_connect('draw_event', self._on_draw)
        self._canvas.mpl_connect('button_press_event',
                                 self._on_button_press)
        self._canvas.mpl_connect('button_release_event',
                                 self._on_button_release)
        self._canvas.mpl_connect('key_press_event',
                                 self._on_key_press)
        self._canvas.mpl_connect('motion_notify_event',
                                 self._on_motion_notify)

        self._canvas.draw()
        plt.show()

    def _get_ind_under_point(self, event):
        """
        get the index of the vertex under point if within epsilon tolerance
        """
        xy = self._line.get_xydata()
        xy = self._line.get_transform().transform(xy)
        x, y = xy[:, 0], xy[:, 1]

        d = np.hypot(x - event.x, y - event.y)
        ind = d.argmin()

        if d[ind] >= self._epsilon:
            ind = None
        return ind

    def _get_coords(self, event, sign='+'):
        """
        Get mouse click coordinates
        """
        x, y = event.xdata, event.ydata
        d = np.hypot(self._xc - x, self._yc - y)
        idx = np.argmin(d)
        i, j = np.unravel_index(idx, d.shape)

        # Print coordiante to screen
        if self.hgrid.spherical:
            lon, lat = self._proj_az(x, y, inverse=True)
            print(r'      ' + sign + '%5d %5d %12.1f %12.1f %10.3f %10.3f'
                  % (j, i, x, y, lon, lat))
            return j, i, x, y, lon, lat
        else:
            print(r'      ' + sign + '%5d %5d %12.1f %12.1f'
                  % (j, i, x, y))
            return j, i, x, y

    def _on_draw(self, event):
        self._bbox = self._canvas.copy_from_bbox(self._ax.bbox)
        self._ax.draw_artist(self._line)
        return

    def _on_button_press(self, event):
        if not self._clicking:
            return
        if event.inaxes is None:
            return

        if event.button == 1:
            # Process track coordinates
            npts = len(self._line.get_xydata())
            if npts < 2:
                # If less than two points, always insert a new point
                if npts == 0:
                    iidx = 0
                elif npts == 1:
                    iidx = 1
            else:
                # If more than two points, add a point to an existing line
                iidx = 0
                self._ind = self._get_ind_under_point(event)
                if self._ind is None:
                    p = event.x, event.y
                    xy = self._line.get_xydata()
                    xy_ax = self._line.get_transform().transform(xy)
                    for i in range(npts-1):
                        d = dist_point_to_segment(p, xy_ax[i], xy_ax[i+1])
                        if d <= self._epsilon:
                            iidx = i+1
                            break
                    if iidx == 0:
                        d0 = (xy_ax[0, 0] - event.x)**2 + \
                             (xy_ax[0, 1] - event.y)**2
                        d1 = (xy_ax[-1, 0] - event.x)**2 + \
                             (xy_ax[-1, 1] - event.y)**2
                        if d1 < d0:
                            iidx = npts

            if self._ind is None:
                # Add a point in the middle of the track line
                xy = self._line.get_xydata()
                xy = np.insert(xy, iidx, [event.xdata, event.ydata], axis=0)
                self._line.set_data(xy[:, 0], xy[:, 1])
                if self.hgrid.spherical:
                    xi, eta, x, y, lon, lat = self._get_coords(event)
                else:
                    xi, eta, x, y = self._get_coords(event)
                self.xi.insert(iidx, xi)
                self.eta.insert(iidx, eta)
                self.x.insert(iidx, x)
                self.y.insert(iidx, y)
                self.angle.insert(iidx, self.hgrid.angle_rho[eta, xi])
                if self.hgrid.spherical:
                    self.lon.insert(iidx, lon)
                    self.lat.insert(iidx, lat)

        elif event.button == 3:
            # Delete a point from the track
            ind = self._get_ind_under_point(event)
            if ind is not None:
                self._get_coords(event, sign='-')
                xy = np.asarray(self._line.get_xydata())
                xy = np.delete(xy, ind, axis=0)
                self._line.set_data(xy[:, 0], xy[:, 1])
                self.xi.pop(ind)
                self.eta.pop(ind)
                self.x.pop(ind)
                self.y.pop(ind)
                self.angle.pop(ind)
                if self.hgrid.spherical:
                    self.lon.pop(ind)
                    self.lat.pop(ind)

        if self._line.stale:
            self._canvas.draw_idle()

    def _on_button_release(self, event):
        if not self._clicking:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self._ind is None:
            return

        if self.hgrid.spherical:
            xi, eta, x, y, lon, lat = self._get_coords(event, sign='>')
        else:
            xi, eta, x, y = self._get_coords(event)
        self.xi[self._ind], self.eta[self._ind] = xi, eta
        self.x[self._ind], self.y[self._ind] = x, y
        self.angle[self._ind] = self.hgrid.angle_rho[eta, xi]
        if self.hgrid.spherical:
            self.lon[self._ind], self.lat[self._ind] = lon, lat
        self._ind = None
        return

    def _on_motion_notify(self, event):
        """
        on mouse movement
        """
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        xy = self._line.get_xydata()
        xy[self._ind] = event.xdata, event.ydata
        self._line.set_data(xy[:, 0], xy[:, 1])

        self._canvas.restore_region(self._bbox)
        self._ax.draw_artist(self._line)
        self._canvas.blit(self._ax.bbox)
        return

    def _on_key_press(self, event):
        if event.key == 'e':
            self._clicking = not self._clicking
            self._ax.set_title(
                'Editing %s -- click "e" to toggle' % self._clicking)
            self._canvas.draw()
            if self._clicking:
                if self.hgrid.spherical:
                    print('Coords:   xi,  eta,           x,           y,' +
                          '       lon,       lat')
                else:
                    print('Coords:   xi,  eta,           x,           y')
        elif event.key == 'N':
            print('Ending editing...')
            plt.close()
            return


class TrackSelector(GetPositionFromMap):
    """
    Get cell index position and coordinate Interactively.
    """

    def interp(self, N=100):
        """
        Interpolate the transect on selected nodes onto N+1 points.
        """
        self.xi, self.eta = np.asarray(self.xi), np.asarray(self.eta)
        self.x, self.y = np.asarray(self.x), np.asarray(self.y)
        self.angle = np.asarray(self.angle)
        a0 = np.zeros(len(self.x))
        a0[1:] = np.hypot(self.x[1:] - self.x[:-1],
                          self.y[1:] - self.y[:-1]).cumsum()
        a1 = np.linspace(a0[0], a0[-1], N+1)
        a0m, a1m = np.meshgrid(a0, a1)
        dis = a0m - a1m
        dis = dis[:, 1:]*dis[:, :-1]
        idx = dis.argmin(axis=1)
        frac = (a1 - a0[idx])/(a0[idx+1] - a0[idx])

        x = self.x[idx] + frac*(self.x[idx+1] - self.x[idx])
        y = self.y[idx] + frac*(self.y[idx+1] - self.y[idx])
        xi, eta = [], []
        for xx, yy in zip(x, y):
            d = np.hypot(self._xc - xx, self._yc - yy)
            idx = d.argmin()
            etai, xii = np.unravel_index(idx, d.shape)
            xi.append(xii)
            eta.append(etai)
        xi, eta = np.array(xi), np.array(eta)
        angle = self.hgrid.angle_rho[eta, xi]

        self.dis0, self.dis = a0, a1
        self.xi0, self.eta0 = self.xi, self.eta
        self.x0, self.y0, self.angle0 = self.x, self.y, self.angle
        self.xi, self.eta, self.x, self.y, self.angle = xi, eta, x, y, angle
        if self.hgrid.spherical:
            lon, lat = self._proj_az(x, y, inverse=True)
            self.lon0, self.lat0 = self.lon, self.lat
            self.lon, self.lat = lon, lat
        return


def rho_to_vert(xr: np.ndarray, yr: np.ndarray,
                pm: np.ndarray, pn: np.ndarray,
                ang: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate x/y coordinates on vertices.
    """

    Mp, Lp = xr.shape
    x = np.empty((Mp+1, Lp+1), dtype='d')
    y = np.empty((Mp+1, Lp+1), dtype='d')
    x[1:-1, 1:-1] = 0.25*(xr[1:, 1:]+xr[1:, :-1]+xr[:-1, 1:]+xr[:-1, :-1])
    y[1:-1, 1:-1] = 0.25*(yr[1:, 1:]+yr[1:, :-1]+yr[:-1, 1:]+yr[:-1, :-1])

    # east side
    theta = 0.5*(ang[:-1, -1]+ang[1:, -1])
    dx = 0.5*(1.0/pm[:-1, -1]+1.0/pm[1:, -1])
    dy = 0.5*(1.0/pn[:-1, -1]+1.0/pn[1:, -1])
    x[1:-1, -1] = x[1:-1, -2] + dx*np.cos(theta)
    y[1:-1, -1] = y[1:-1, -2] + dx*np.sin(theta)

    # west side
    theta = 0.5*(ang[:-1, 0]+ang[1:, 0])
    dx = 0.5*(1.0/pm[:-1, 0]+1.0/pm[1:, 0])
    dy = 0.5*(1.0/pn[:-1, 0]+1.0/pn[1:, 0])
    x[1:-1, 0] = x[1:-1, 1] - dx*np.cos(theta)
    y[1:-1, 0] = y[1:-1, 1] - dx*np.sin(theta)

    # north side
    theta = 0.5*(ang[-1, :-1]+ang[-1, 1:])
    dx = 0.5*(1.0/pm[-1, :-1]+1.0/pm[-1, 1:])
    dy = 0.5*(1.0/pn[-1, :-1]+1.0/pn[-1, 1:])
    x[-1, 1:-1] = x[-2, 1:-1] - dy*np.sin(theta)
    y[-1, 1:-1] = y[-2, 1:-1] + dy*np.cos(theta)

    # here we are now going to the south side..
    theta = 0.5*(ang[0, :-1]+ang[0, 1:])
    dx = 0.5*(1.0/pm[0, :-1]+1.0/pm[0, 1:])
    dy = 0.5*(1.0/pn[0, :-1]+1.0/pn[0, 1:])
    x[0, 1:-1] = x[1, 1:-1] + dy*np.sin(theta)
    y[0, 1:-1] = y[1, 1:-1] - dy*np.cos(theta)

    # Corners
    x[0, 0] = 4.0*xr[0, 0]-x[1, 0]-x[0, 1]-x[1, 1]
    x[-1, 0] = 4.0*xr[-1, 0]-x[-2, 0]-x[-1, 1]-x[-2, 1]
    x[0, -1] = 4.0*xr[0, -1]-x[0, -2]-x[1, -1]-x[1, -2]
    x[-1, -1] = 4.0*xr[-1, -1]-x[-2, -2]-x[-2, -1]-x[-1, -2]

    y[0, 0] = 4.0*yr[0, 0]-y[1, 0]-y[0, 1]-y[1, 1]
    y[-1, 0] = 4.0*yr[-1, 0]-y[-2, 0]-y[-1, 1]-y[-2, 1]
    y[0, -1] = 4.0*yr[0, -1]-y[0, -2]-y[1, -1]-y[1, -2]
    y[-1, -1] = 4.0*yr[-1, -1]-y[-2, -2]-y[-2, -1]-y[-1, -2]

    return x, y


def rho_to_vert_geo(lonr: np.ndarray, latr: np.ndarray,
                    lonp: np.ndarray, latp: np.ndarray,
                    proj: Union[type(None), pyproj.Proj] = None) -> \
                        (np.ndarray, np.ndarray):
    """
    Calculate lon/lat coordinates on vertices.
    """

    if proj is not None:
        xr, yr = proj(lonr, latr)
        xp, yp = proj(lonp, latp)
    else:
        xr, yr = lonr, latr
        xp, yp = lonp, latp

    Mm, Lm = xr.shape
    x = np.zeros((Mm+1, Lm+1))
    y = np.zeros((Mm+1, Lm+1))

    x[1:-1, 1:-1] = xp[:, :]
    y[1:-1, 1:-1] = yp[:, :]

    # Edges
    x[0, 1:-1] = xr[0, 1:] + xr[0, :-1] - xp[0, :]
    y[0, 1:-1] = yr[0, 1:] + yr[0, :-1] - yp[0, :]
    x[-1, 1:-1] = xr[-1, 1:] + xr[-1, :-1] - xp[-1, :]
    y[-1, 1:-1] = yr[-1, 1:] + yr[-1, :-1] - yp[-1, :]
    x[1:-1, 0] = xr[1:, 0] + xr[:-1, 0] - xp[:, 0]
    y[1:-1, 0] = yr[1:, 0] + yr[:-1, 0] - yp[:, 0]
    x[1:-1, -1] = xr[1:, -1] + xr[:-1, -1] - xp[:, -1]
    y[1:-1, -1] = yr[1:, -1] + yr[:-1, -1] - yp[:, -1]

    # Corners
    x[0, 0] = 4*xr[0, 0] - x[0, 1] - x[1, 0] - x[1, 1]
    y[0, 0] = 4*yr[0, 0] - y[0, 1] - y[1, 0] - y[1, 1]
    x[-1, 0] = 4*xr[-1, 0] - x[-1, 1] - x[-2, 0] - x[-2, 1]
    y[-1, 0] = 4*yr[-1, 0] - y[-1, 1] - y[-2, 0] - y[-2, 1]
    x[0, -1] = 4*xr[0, -1] - x[1, -1] - x[0, -2] - x[1, -2]
    y[0, -1] = 4*yr[0, -1] - y[1, -1] - y[0, -2] - y[1, -2]
    x[-1, -1] = 4*xr[-1, -1] - x[-1, -2] - x[-2, -1] - x[-2, -2]
    y[-1, -1] = 4*yr[-1, -1] - y[-1, -2] - y[-2, -1] - y[-2, -2]

    if proj is not None:
        lon, lat = proj(x, y, inverse=True)
    else:
        lon, lat = x, y

    return lon, lat


def uvp_masks(rmask: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    return u-, v-, and psi-masks based on input rho-mask

    Parameters
    ----------

    rmask : ndarray
        mask at CGrid rho-points

    Returns
    -------
    (umask, vmask, pmask) : ndarrays
        masks at u-, v-, and psi-points
    """

    rmask = np.asarray(rmask)
    assert rmask.ndim == 2, 'rmask must be a 2D array'
    assert np.all((rmask == 0) | (rmask == 1)), \
        'rmask array must contain only ones and zeros.'

    umask = rmask[:, :-1] * rmask[:, 1:]
    vmask = rmask[:-1, :] * rmask[1:, :]
    pmask = rmask[:-1, :-1] * rmask[:-1, 1:] * rmask[1:, :-1] * rmask[1:, 1:]

    return umask, vmask, pmask


if __name__ == '__main__':
    geographic = False
    shp = (32, 32)
    if geographic:
        lon = (-71.977385177601761, -70.19173825913137,
               -63.045075098584945, -64.70104074097425)
        lat = (42.88215610827428, 41.056141745853786,
               44.456701607935841, 46.271758064353897)
        beta = [1.0, 1.0, 1.0, 1.0]

        mproj = ccrs.Stereographic(
            central_latitude=lat.mean(),
            central_longitude=lon.mean(),
            false_easting=0.0, false_northing=0.0)
        proj = pyproj.Proj(mproj.proj4_init)
        grd = gridgen(lon, lat, beta, shp, proj=proj)
        grd.mask_land()

        fig, ax = plt.subplots(subplot_kw={'projection': mproj})
        ax.pcolormesh(grd.x, grd.y, grd.mask, edgecolor='gray')
        ax.coastlines(resolution='10m')
        plt.show()
    else:
        x = [0.2, 0.85, 0.9, 0.82, 0.23]
        y = [0.2, 0.25, 0.5, 0.82, .83]
        beta = [1.0, 1.0, 0.0, 1.0, 1.0]

        grd = gridgen(x, y, beta, shp, proj=None)

        ax = plt.subplot()
        BoundaryInteractor(x, y, beta, proj=None)
        plt.show()
