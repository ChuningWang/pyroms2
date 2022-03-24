"""
Tools for creating and working with Line (Station) Grids
"""

from typing import Union
import pyproj
import numpy as np


_atype = Union[type(None), np.ndarray]
_ptype = Union[type(None), pyproj.Proj]


class StaHGrid:
    """
    Stations Grid

    EXAMPLES:
    --------

    >>> x = arange(8)
    >>> y = arange(8)*2-1
    >>> grd = pyroms.grid.StaHGrid(x, y)
    >>> print grd.x
    [4.5 4.5 4.5 4.5 4.5 4.5 4.5]
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, angle: _atype = None):

        assert x.ndim == 1 and y.ndim == 1 and x.shape == y.shape, \
               'x and y must be 2D arrays of the same size.'

        mask = np.isnan(x) | np.isnan(y)
        if np.any(mask):
            x = np.ma.masked_where(mask, x)
            y = np.ma.masked_where(mask, y)

        self.spherical = False
        self._x, self._y = x, y
        if angle is None:
            self.angle = np.zeros(len(self.y))
        else:
            self.angle = angle
        return

    x = property(lambda self: self._x)
    y = property(lambda self: self._y)


class StaHGridGeo(StaHGrid):
    """
    Stations Grid

    EXAMPLES:
    --------

    >>> lon = arange(8)
    >>> lat = arange(8)*2-1
    >>> proj = pyproj()
    >>> grd = pyroms.grid.StaHGridGeo(lon, lat, proj)
    >>> print grd.x
    [xxx, xxx, xxx, xxx, xxx, xxx, xxx, xxx]
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray,
                 x: _atype = None, y: _atype = None,
                 angle: _atype = None, proj: _ptype = None):

        self.spherical = True
        self._lon, self._lat = lon, lat
        self.proj = proj
        if x is not None and y is not None:
            super(StaHGridGeo, self).__init__(x, y, angle)
            self.spherical = True
        else:
            if proj is not None:
                self._x, self._y = proj(lon, lat)
            else:
                raise ValueError('Projection transformer must be ' +
                                 'provided if x/y are missing.')
        return

    @property
    def lon(self):
        return self._lon

    @lon.setter
    def lon(self, lon):
        if self.proj is not None:
            self.__init__(lon, self._lat, angle=self.angle, proj=self.proj)
        else:
            self._lon = lon

    @property
    def lat(self):
        return self._lat

    @lat.setter
    def lat(self, lat):
        if self.proj is not None:
            self.__init__(self._lon, lat, angle=self.angle, proj=self.proj)
        else:
            self._lat = lat
