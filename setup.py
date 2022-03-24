"""
python tools for working with ROMS.

Requires:
    NumPy (http://numpy.scipy.org)
    matplotlib with the Basemap toolkit (http://matplotlib.sourceforge.net)
    netCDF4 (http://code.google.com/p/netcdf4-python/)

Contains:
    hgrid  -  Tools for dealing with curvilinear grids
      BoundaryInteractor
      _Focus_x
      _Focus_y
       Focus
      CGrid
      CGrid_geo
      Gridgen
      edit_mask_mesh
      get_position_from_map

    vgrid  -  Various vertical coordinates
      s_coordinate
      z_r
      z_w
      z_coordinate

    grid  -  ROMS Grid object
      ROMS_Grid
      ROMS_gridinfo
      print_ROMS_gridinfo
      list_ROMS_gridid
      get_ROMS_hgrid
      get_ROMS_vgrid
      get_ROMS_grid
      write_ROMS_grid

    io  -  wrapper for netCDF4
      Dataset
      MFDataset

    cf  -  CF compliant files tools
      time

    utility  -  Some basic tools
      get_lonlat
      get_ij
      roms_varlist
      get_roms_var
      get_bottom
      get_surface
"""

from setuptools import setup, find_packages

doclines = __doc__.split("\n")

setup(
    name="pyroms2",
    version="0.1.0",
    author="Chuning Wang",
    author_email="wangchuning@sjtu.edu.cn",
    description=doclines[0],
    url="https://github.com/ChuningWang/pyroms2",
    license="MIT",
    platforms=["Any"],
    packages=find_packages(),

    classifiers=[
        "Development Status :: alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: MIT",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Python Modules",
        "Topic :: Libraries",
    ],

    python_requires=">=3.9",
    install_requires=["xesmf >= 0.6.0",
                      "cartopy >= 0.20.0",
                      "xgcm >= 0.6.0",
                      "netcdf4 >= 1.5.5",
                     ],
    setup_requires=[],
    tests_requires=[],
    dependency_links=[],
)
