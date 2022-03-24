"""
This code is adapted from the matlab code 'LP Bathymetry' by Mathieu
  Dutour Sikiric. For a description of the method, see

M. Dutour Sikiric, I. Janekovic, M. Kuzmic, A new approach to bathymetry
  smoothing in sigma-coordinate ocean models, Ocean Modelling 29 (2009)
  128--136.

and

http://drobilica.irb.hr/~mathieu/Bathymetry/index.html
"""

import numpy as np
import xarray as xr


def roughness0(hraw, mask):
    """
    Usage:
        rx0 = RoughnessMatrix(hraw, mask)

    Inputs:
        hraw - numpy array or Xarray, water depth (column thickness)
        mask - numpy array or Xarray, maks of grid

    Output:
        rx0 - numpy array or Xarray, roughness index rx0 values
    """

    eta_rho, xi_rho = hraw.shape

    h = np.ma.masked_where(mask == 0, hraw)
    rx0 = np.ma.zeros((4, eta_rho, xi_rho))
    rx0[0, 1:-1, 1:-1] = np.abs(
        (h[1:-1, 2:]-h[1:-1, 1:-1])/(h[1:-1, 2:]+h[1:-1, 1:-1]))
    rx0[1, 1:-1, 1:-1] = np.abs(
        (h[1:-1, :-2]-h[1:-1, 1:-1])/(h[1:-1, :-2]+h[1:-1, 1:-1]))
    rx0[2, 1:-1, 1:-1] = np.abs(
        (h[2:, 1:-1]-h[1:-1, 1:-1])/(h[2:, 1:-1]+h[1:-1, 1:-1]))
    rx0[3, 1:-1, 1:-1] = np.abs(
        (h[:-2, 1:-1]-h[1:-1, 1:-1])/(h[:-2, 1:-1]+h[1:-1, 1:-1]))

    rx0 = rx0.max(axis=0)
    rx0[rx0.mask] = 0
    rx0 = rx0.data

    if isinstance(hraw, xr.DataArray):
        rx0 = xr.DataArray(rx0, hraw.coords)

    return rx0
