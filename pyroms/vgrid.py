"""
Various vertical coordinates

Presently, only ocean s-coordinates are supported. Future plans will be to
include all of the vertical coordinate systems defined by the CF conventions.
"""

from typing import Union
import numpy as np


class SCoord:
    """
    ROMS vertical s-coordinate generater.

    SCoord generates a ROMS vertical grid class that contains information
    of grid transformation factor.

    Usage:
        vgrid = SCoord(h, theta_b, theta_s, Tcline, N,
                       Vtrans=2, Vstretch=4,
                       hraw=None, zeta=None, zice=None)
    Inputs:
        h        float[Nxy]    - grid bathymetry (depth)
        theta_b  float         - bottom boundary layer intensification factor
        theta_s  float         - surface boundary layer intensification factor
        Tcline   float         - width of surface boundary layer
        N        int           - number of rho-layers
        Vtrans   int           - transformation function ID number
        Vstretch int           - stretching function ID number
        hraw     int[Nxy]      - raw bathymetry
        zeta     int[[Nt,Nxy]] - sea surface height
        zice     int[[Nt,Nxy]] - iceshelf draft depth
    Outputs:
        vgrid     Class            - vertical grid
        vgrid.z_r Class, indexable - depth at rho points
        vgrid.z_w Class, indexable - depth at w points

    Only these combinations of vertical transformation and stretching
    functions are acceptable:
        Vtrans = 1, Vstretch = 1:
            Song and Haidvogel (1994) vertical coordinate transformation
            and stretching functions.
        Vtrans = 2, Vstretch = 2:
            A. Shchepetkin (2005) UCLA-ROMS vertical coordinate
            transformation and stretching functions.
        Vtrans = 2, Vstretch = 4 (default):
            A. Shchepetkin (2010) UCLA-ROMS vertical coordinate
            transformation and stretching functions.
        Vtrans = 2, Vstretch = 5:
            Souza et al. (2015) quadratic Legendre polynomial vertical
            coordinate transformation and stretching functions

    For more information check:
        https://www.myroms.org/wiki/Vertical_S-coordinate
    """
    def __init__(self,
                 h: np.ndarray,
                 theta_b: float, theta_s: float, Tcline: float, N: int,
                 Vtrans: int = 2, Vstretch: int = 4,
                 hraw: Union[type(None), np.ndarray] = None,
                 zeta: Union[type(None), np.ndarray] = None,
                 zice: Union[type(None), np.ndarray] = None,
                 verbose: bool = True):

        self._h = h
        self._theta_b = theta_b
        self._theta_s = theta_s
        self._Tcline = Tcline
        self._N = int(N)

        self._Vtrans = Vtrans
        self._Vstretch = Vstretch

        if verbose:
            print('Constructing Vertical Grid.')
        if self.Vtrans == 1 and self.Vstretch == 1:
            if verbose:
                print('Song and Haidvogel (1994) vertical coordinate ' +
                      'transformation (Vtransform=1) and ' +
                      'stretching (Vstretching=1) functions.')
        elif self.Vtrans == 2 and self.Vstretch == 2:
            if verbose:
                print('A. Shchepetkin (2005) UCLA-ROMS vertical coordinate ' +
                      'transformation (Vtransform=2) and ' +
                      'stretching (Vstretching=2) functions.')
        elif self.Vtrans == 2 and self.Vstretch == 4:
            if verbose:
                print('A. Shchepetkin (2010) UCLA-ROMS vertical coordinate ' +
                      'transformation (Vtransform=2) and ' +
                      'stretching (Vstretching=4) functions.')
        elif self.Vtrans == 2 and self.Vstretch == 5:
            if verbose:
                print('Souza et al. (2015) quadratic Legendre polynomial ' +
                      'vertical coordinate transformation (Vtransform=2) and' +
                      'stretching (Vstretching=4) functions')
        else:
            raise ValueError(
                'Unknown combination of ' +
                'transformation (Vtransform=%01d) ' % self.Vtrans + 'and ' +
                'stretching (Vstretching=%01d) ' % self.Vstretch +
                'functions.')

        self.hmin = self.h.min()
        self.Np = self.N+1

        if self.Vtrans == 1:
            self.hc = min(self.hmin, self.Tcline)
            if (self.Tcline > self.hmin):
                print(
                    'Vertical transformation parameters are not defined ' +
                    'correctly in either gridid.txt or in the history files:' +
                    '\n' +
                    'Tcline = %d and hmin = %d.' % (self.Tcline, self.hmin) +
                    '\n' +
                    'You need to make sure that Tcline <= hmin when using ' +
                    'transformation 1.')
        else:
            self.hc = self.Tcline

        if self.Vtrans == 2 and self.Vstretch == 2:
            self.Aweight = 1.0
            self.Bweight = 1.0

        if hraw is None:
            self.hraw = h
        else:
            self.hraw = hraw
        if zeta is None:
            self._zeta = np.zeros(self._h.shape)
        else:
            self._zeta = zeta

        if zice is None:
            self._zice = np.zeros(self._h.shape)
        else:
            self._zice = np.abs(zice)

        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        # Calculate s_rho and s_w
        if self.Vstretch == 5:
            lev = np.arange(1, self.N+1) - .5
            self.s_r = -(lev*lev-2*lev*self.N+lev+self.N*self.N-self.N) / \
                (1.0 * self.N * self.N - self.N) - \
                0.01 * (lev * lev - lev * self.N) / (1.0 - self.N)
        else:
            lev = np.arange(1, self.N+1, 1)
            ds = 1.0 / self.N
            self.s_r = -self.c1 + (lev - self.p5) * ds

        if self.Vstretch == 5:
            lev = np.arange(0, self.Np, 1)
            self.s_w = -(lev*lev-2*lev*self.N+lev+self.N*self.N-self.N) / \
                (self.N * self.N - self.N) - \
                0.01 * (lev * lev - lev * self.N) / (self.c1 - self.N)
        else:
            lev = np.arange(0, self.Np, 1)
            ds = 1.0 / (self.Np-1)
            self.s_w = -self.c1 + lev * ds

        # Calculate Cs_r and Cs_w
        if self.Vtrans == 1 and self.Vstretch == 1:
            if (self.theta_s >= 0):
                Ptheta = np.sinh(self.theta_s * self.s_r) / \
                         np.sinh(self.theta_s)
                Rtheta = np.tanh(self.theta_s * (self.s_r + self.p5)) / \
                    (self.c2 * np.tanh(self.p5 * self.theta_s)) - self.p5
                self.Cs_r = (self.c1-self.theta_b)*Ptheta + self.theta_b*Rtheta
            else:
                self.Cs_r = self.s_r
        elif self.Vtrans == 2 and self.Vstretch == 2:
            if (self.theta_s >= 0):
                Csur = (self.c1 - np.cosh(self.theta_s * self.s_r)) / \
                         (np.cosh(self.theta_s) - self.c1)
                if (self.theta_b >= 0):
                    Cbot = np.sinh(self.theta_b * (self.s_r + self.c1)) / \
                           np.sinh(self.theta_b) - self.c1
                    Cweight = (self.s_r + self.c1)**self.Aweight * \
                              (self.c1 + (self.Aweight / self.Bweight) *
                               (self.c1 - (self.s_r+self.c1)**self.Bweight))
                    self.Cs_r = Cweight * Csur + (self.c1 - Cweight) * Cbot
                else:
                    self.Cs_r = Csur
            else:
                self.Cs_r = self.s_r
        elif self.Vtrans == 2 and self.Vstretch == 4:
            if (self.theta_s > 0):
                Csur = (self.c1 - np.cosh(self.theta_s * self.s_r)) / \
                         (np.cosh(self.theta_s) - self.c1)
            else:
                Csur = -self.s_r**2
            if (self.theta_b > 0):
                Cbot = (np.exp(self.theta_b * Csur) - self.c1) / \
                       (self.c1 - np.exp(-self.theta_b))
                self.Cs_r = Cbot
            else:
                self.Cs_r = Csur
        elif self.Vtrans == 2 and self.Vstretch == 5:
            if self.theta_s > 0:
                csur = (self.c1 - np.cosh(self.theta_s * self.s_r)) / \
                    (np.cosh(self.theta_s) - self.c1)
            else:
                csur = -(self.s_r * self.s_r)
            if self.theta_b > 0:
                self.Cs_r = (np.exp(self.theta_b*(csur+self.c1)) - self.c1) / \
                    (np.exp(self.theta_b) - self.c1) - self.c1
            else:
                self.Cs_r = csur

        if self.Vtrans == 1 and self.Vstretch == 1:
            if (self.theta_s >= 0):
                Ptheta = np.sinh(self.theta_s * self.s_w) / \
                         np.sinh(self.theta_s)
                Rtheta = np.tanh(self.theta_s * (self.s_w + self.p5)) / \
                    (self.c2 * np.tanh(self.p5 * self.theta_s)) - self.p5
                self.Cs_w = (self.c1-self.theta_b)*Ptheta + self.theta_b*Rtheta
            else:
                self.Cs_w = self.s_w
        elif self.Vtrans == 2 and self.Vstretch == 2:
            if (self.theta_s >= 0):
                Csur = (self.c1 - np.cosh(self.theta_s * self.s_w)) / \
                         (np.cosh(self.theta_s) - self.c1)
                if (self.theta_b >= 0):
                    Cbot = np.sinh(self.theta_b * (self.s_w + self.c1)) / \
                           np.sinh(self.theta_b) - self.c1
                    Cweight = (self.s_w + self.c1)**self.Aweight * \
                              (self.c1 + (self.Aweight / self.Bweight) *
                               (self.c1 - (self.s_w + self.c1)**self.Bweight))
                    self.Cs_w = Cweight * Csur + (self.c1 - Cweight) * Cbot
                else:
                    self.Cs_w = Csur
            else:
                self.Cs_w = self.s_w
        elif self.Vtrans == 2 and self.Vstretch == 4:
            if (self.theta_s > 0):
                Csur = (self.c1 - np.cosh(self.theta_s * self.s_w)) / \
                         (np.cosh(self.theta_s) - self.c1)
            else:
                Csur = -self.s_w**2
            if (self.theta_b > 0):
                Cbot = (np.exp(self.theta_b * Csur) - self.c1) / \
                       (self.c1 - np.exp(-self.theta_b))
                self.Cs_w = Cbot
            else:
                self.Cs_w = Csur
        elif self.Vtrans == 2 and self.Vstretch == 5:
            if self.theta_s > 0:
                csur = (self.c1 - np.cosh(self.theta_s * self.s_w)) / \
                    (np.cosh(self.theta_s) - self.c1)
            else:
                csur = -(self.s_w * self.s_w)
            if self.theta_b > 0:
                self.Cs_w = (np.exp(self.theta_b*(csur+self.c1)) - self.c1) / \
                    (np.exp(self.theta_b) - self.c1) - self.c1
            else:
                self.Cs_w = csur

        self.z_r = ZR(self.h, self.hc, self.N, self.s_r, self.Cs_r,
                      self.zeta, self.zice, self.Vtrans)
        self.z_w = ZW(self.h, self.hc, self.Np, self.s_w, self.Cs_w,
                      self.zeta, self.zice, self.Vtrans)
        return

    # Property decorators. Any updates on the initial parameters will make
    # the vgrid to reinitiallize.
    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, val):
        self.__init__(val, self._theta_b, self._theta_s, self._Tcline,
                      self._N, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=self._zeta, zice=self._zice,
                      verbose=False)

    @property
    def theta_b(self):
        return self._theta_b

    @theta_b.setter
    def theta_b(self, val):
        self.__init__(self._h, val, self._theta_s, self._Tcline,
                      self._N, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=self._zeta, zice=self._zice,
                      verbose=False)

    @property
    def theta_s(self):
        return self._theta_s

    @theta_s.setter
    def theta_s(self, val):
        self.__init__(self._h, self._theta_b, val, self._Tcline,
                      self._N, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=self._zeta, zice=self._zice,
                      verbose=False)

    @property
    def Tcline(self):
        return self._Tcline

    @Tcline.setter
    def Tcline(self, val):
        self.__init__(self._h, self._theta_b, self._theta_s, val,
                      self._N, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=self._zeta, zice=self._zice,
                      verbose=False)

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, val):
        self.__init__(self._h, self._theta_b, self._theta_s, self._Tcline,
                      val, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=self._zeta, zice=self._zice,
                      verbose=False)

    @property
    def Vtrans(self):
        return self._Vtrans

    @Vtrans.setter
    def Vtrans(self, val):
        self.__init__(self._h, self._theta_b, self._theta_s, self._Tcline,
                      self._N, Vtrans=val, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=self._zeta, zice=self._zice)

    @property
    def Vstretch(self):
        return self._Vstretch

    @Vstretch.setter
    def Vstretch(self, val):
        self.__init__(self._h, self._theta_b, self._theta_s, self._Tcline,
                      self._N, Vtrans=self._Vtrans, Vstretch=val,
                      hraw=self.hraw, zeta=self._zeta, zice=self._zice)

    @property
    def zeta(self):
        return self._zeta

    @zeta.setter
    def zeta(self, val):
        self.__init__(self._h, self._theta_b, self._theta_s, self._Tcline,
                      self._N, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=val, zice=self._zice,
                      verbose=False)
        return

    @zeta.deleter
    def zeta(self):
        self.__init__(self._h, self._theta_b, self._theta_s, self._Tcline,
                      self._N, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=None, zice=self._zice,
                      verbose=False)
        return

    @property
    def zice(self):
        return self._zice

    @zice.setter
    def zice(self, val):
        self.__init__(self._h, self._theta_b, self._theta_s, self._Tcline,
                      self._N, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=self._zeta, zice=val,
                      verbose=False)
        return

    @zice.deleter
    def zice(self):
        self.__init__(self._h, self._theta_b, self._theta_s, self._Tcline,
                      self._N, Vtrans=self._Vtrans, Vstretch=self._Vstretch,
                      hraw=self.hraw, zeta=self._zeta, zice=None,
                      verbose=False)
        return

    @property
    def dz(self):
        return np.diff(self.z_w[:], axis=-(self.h.ndim + 1))


class Z:
    def __init__(self,
                 h: np.ndarray, hc: float, N: int,
                 s: np.ndarray, Cs: np.ndarray,
                 zeta: np.ndarray, zice: np.ndarray,
                 Vtrans: int):
        self.h = h
        self.hc = hc
        self.N = N
        self.s = s
        self.Cs = Cs
        self.zeta = zeta
        self.zice = zice
        self.Vtrans = Vtrans

        return

    def __getitem__(self, key):
        # read in dimension of arrays for further use
        ztdim = self.zeta.ndim
        zidim = self.zice.ndim
        hdim = self.h.ndim

        is_tseries = False

        if ztdim == hdim:

            # zeta does not have time dimension
            keyt = (slice(None),)

            if isinstance(key, tuple):
                keyz = key[0:1]
                keyxy = key[1:]
            else:
                keyz = (key,)
                keyxy = ()

        else:

            # zeta has time dimension
            if isinstance(key, tuple):
                keyt = key[0:1]
                keyz = key[1:2]
                keyxy = key[2:]
            else:
                keyt = (key,)
                keyz = ()
                keyxy = ()

            # check if calculate time series
            if isinstance(keyt[0], slice):
                is_tseries = True

        # read in zeta, zice and h. Note the difference of their dims.
        if ztdim == hdim:
            zeta = self.zeta[keyxy]
        else:
            zeta = self.zeta[keyt + keyxy]

        if zidim == hdim:
            zice = self.zice[keyxy]
        else:
            zice = self.zice[keyt + keyxy]

        h = self.h[keyxy]

        # if zeta or zice is a single number, make it an array for indexing.
        if type(zeta) not in [np.ndarray, np.ma.MaskedArray]:
            zeta = np.array([zeta])
        if type(zice) not in [np.ndarray, np.ma.MaskedArray]:
            zice = np.array([zice])

        if is_tseries:
            # if zeta or zice is time series, expand the other one in time dim.
            ti = zeta.shape[0]
            if ztdim > zidim:
                rep = (ti,)
                for i in range(zice.ndim):
                    rep = rep + (1,)
                zice = np.tile(zice, rep)
        else:
            # otherwise expand time dim of both.
            ti = 1
            zeta = zeta[np.newaxis, :]
            zice = zice[np.newaxis, :]

        # calculate z
        z = np.empty((ti, self.N) + h.shape, 'd')
        if self.Vtrans == 1:
            for n in range(ti):
                for k in range(self.N):
                    z0 = self.hc*self.s[k] + \
                         (h-zice[n]-self.hc)*self.Cs[k]
                    z[n, k] = z0+zeta[n]*(1.0+z0/(h-zice[n]))-zice[n]
        elif self.Vtrans == 2:
            for n in range(ti):
                for k in range(self.N):
                    z0 = (self.hc*self.s[k] +
                          (h-zice[n])*self.Cs[k]) / \
                         (self.hc+h-zice[n])
                    z[n, k] = zeta[n]+(zeta[n]+h-zice[n])*z0-zice[n]

        if is_tseries:
            return z[(slice(None),) + keyz]
        else:
            return z[(0,) + keyz]


class ZR(Z):
    @property
    def s_r(self):
        return self.s

    @property
    def Cs_r(self):
        return self.Cs


class ZW(Z):
    @property
    def s_w(self):
        return self.s

    @property
    def Cs_w(self):
        return self.Cs


class ZCoord:
    """
    return an object that can be indexed to return depths

    z = z_coordinate(h, depth, N)
    """

    def __init__(self, h: np.ndarray, depth: np.ndarray, N: int):
        self.h = np.asarray(h)
        self.N = int(N)

        ndim = len(h.shape)

        if ndim == 2:
            Mm, Lm = h.shape
            self.z = np.zeros((N, Mm, Lm))
        elif ndim == 1:
            Sm = h.shape[0]
            self.z = np.zeros((N, Sm))

        for k in range(N):
            self.z[k, :] = depth[k]
