# pyROMS2

pyROMS2 is a collection of tools to process input and output files for the Regional Ocean Modeling System, [ROMS].
It was originally started by Rob Hetland as a googlecode project, then he morphed it into octant, also hosted on googlecode.
Frederic Castruccio then created a fork and renamed it back to pyROMS, and Kate Hedstrom is the current maintainer.

The offical pyROMS project is now hosted on GitHub.com in the [ESMG/pyroms] repository, default on the [python3] branch.
This branch is originally forked from [ESMG/pyroms], and then I developed it into a separate project hosted on [ChuningWang/pyroms2].
In this branch, many of the computation kernels are rewritten with modern Python packages, such as [Cartopy] and [Xarray], and many new features are added to satisfy my own research needs (such as adding iceshelf).

pyROMS2 provides a series of horizontal and vertical grid utilities, netCDF4 I/O utilities, regridding utilities and an interface to work with Xarray and Xgcm. Its basic functions include:

  * Making Cartisian/Geographic Arakawa C-Grid (with or without an interactive GUI)
  * Making vertical S-Coordinate grid and transformation functions
  * Auto masking land/iceshelf points based on [Natural Earth] geographical datasets
  * Bathymetry smoothing using Scipy's Linear Programming computation kernel
  * Regridding from ocean reanalysis products to a ROMS grid using xESMF
  * Easy interpolation/computation/plotting utilities using Xarray and Xgcm
  * ...

## Compared with [ESMG/pyroms]
Kate's pyROMS branch has been out there for years, and is widely used in the ROMS python community. However, after Python 3.9 being released, [Basemap] is no longer supported and a lot of functions in pyROMS fail to work.
This branch replaced Basemap with [Cartopy], and spent a lot of efforts to enhance the code's performance. Most loops are vectorized, redundant codes are removed.
The implementation of Scipy's Linear Programming functions makes bathymetry smoothing much easier and faster.
The implementation of xESMF makes regridding easier and faster, which faciliates initial/boundary condition generation.

## Compared with [XROMS]
This project has some overlap with [XROMS], which is another excellent python package that integrates [Xarray]/[Xgcm]/[cf-Xarray] interface to work with ROMS input/output files.
But what troubles me most about XROMS is its deep integration with Xgcm and cf-Xarray, which to me is not the best strategy.
[cf-Xarray] provides some generic functions to convert model-specific coordinate/variable names (for ROMS, ETA, XI, RHO/W, x/y, lon/lat, etc.) to [CF Metadata Convention] names, which works best for people familiar with the CF Convention; for ROMS modelers like myself, I still prefer to work with ROMS's original convention and not ready to move on. Besides, the functions inherited from Kate's pyROMS uses ROMS's original convention, porting them to cf-Xarray requires a lot of work.
Another issue I found with cf-Xarray is that during the coordinate/variable name conversion process, cf-Xarray 'guesses' new coordinate/variable names by matching keywords with variable names/attributes, and sometimes this process is not very accurate based on my own experience. What I suggest is to use ROMS's namelist file, varinfo.yaml (or varinfo.dat for ROMS version under 4.0) to read variable names. I haven't done so yet, but will work on this when I have time.
Same thing with [Xgcm] - It is a generic GCM coordinate constructor, but is not comstomized for ROMS (or other S-Coordinate models in general).
This project attempts to minimize, if not totally discard the usage of Xgcm and cf-Xarray and still realize the same functionalities of XROMS. At this moment, Xgcm is used to vertically interpolate data from S-Coordinate to Z-Coordinate, and cf-Xarray is not used in my code. A few weeks earlier I wrote a new function to do vertical interpolation using numba/numpy to replace the vertical interpolation function of Xgcm, after enough testing I'll make it the default interpolation function; but I will keep Xgcm in the dependence list, because it is a quickly envolving package and I plan to dig deeper into it sometime later.

XROMS also provides some high-level plotting functions, which are reallized with holoviews/geoviews. These functions have not been reallized in pyROMS2. If it is to be developed, it will be likely very similar to XROMS.

## Installation

pyROMS2 is still a bit rough around the edges, particularly for installation.
Recent development has been done in Python environments managed by [Conda].
However pyROMS2 itself cannot yet be installed with Conda.

If you are starting from scratch, I recommend that you install [Anaconda] or [Miniconda]
and create a Python 3 environment (as of March 2022, version 3.8 is your best bet).
You should also consider making conda-forge your default channel. See the [conda-forge tips and tricks] page.

If you don't want to use Conda, that's fine, but you will have to do more installation configuration yourself.

### Prerequisites

The following are required and are all available through [Conda-Forge].

  * Python >= 3.8 (Python 3.8 currently recommended for new environments)
  * [netcdf4]
  * [xESMF]
    * [scipy]
    * [Xarray]
      * [numpy]
      * [numba]
      * [cftime]
      * ...
    * ...
  * [Cartopy]
    * [matplotlib]
      * [shapely]
    * [pyproj]
    * ...
  * [xgcm]
  * [gridgen-c]
  * [pip] (if use pip for package installation)

The following packages are required in the offical branch, but is no longer needed in this branch

  * [SCRIP], which is the Spherical Coordinate Remapping and Interpolation Package.
SCRIP is no longer maintained by its development team.
The Python scrip code (a rather old version) is bundled in the offical branch [here](https://github.com/ESMG/pyroms/tree/python3/pyroms/external/scrip).

  * [Basemap], which is no longer supported in Python 3.9, thus is replaced by its successor [Cartopy].

  * [lpsolve55], which is the linear programmiing solver written in **C**.
[lpsolve55] is used when smoothing bathymetry with the
[LP_smoothing_rx0](https://github.com/ChuningWang/pyroms2/tree/main/pyroms/bathy_tools/lp_smoothing.py) function.
For some unknown reason it is very slow and crashes a lot when tested on a laptop.
For this reason, lpsolve55 is replaced by the [scipy] LP solver in the bathymetry smoothing algorithm.
[scipy] very likely also uses [lpsolve55] somewhere in its computation kernel, but is much better bundled in [scipy].

### Install using PIP

First, install miniconda. Then, I recommend creating a new environment and install the following dependence

```bash
# Create a conda environment for pyROMS2
$ conda create -n pyroms_env python=3.8
$ conda activate pyroms_env
$ conda install -c conda-forge xesmf
$ conda install -c conda-forge cartopy
$ conda install -c conda-forge xgcm
$ conda install -c conda-forge netcdf4
$ conda install -c conda-forge gridgen
```

To clone a copy of the source and install the pyROMS packages, you can use the following commands
```bash
# Install pyROMS2
$ cd /path/to/install
$ git clone https://github.com/ChuningWang/pyroms2.git
$ pip install -e ./pyroms2
```

An [editable-mode](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)
installation is recommended becauses it means changes you make to your copy of the source code will take effect when you import the modules.
If you don't want this you can omit the "-e" option.

The "pip install" command runs "python setup.py install" (or "python setup.py develop" with the "-e" switch) in each of the subdirectories listed.
The "pip install" form is recommended because it allow easy removal (see below)

The above should work on most Linuces and on OSX with the system gcc and gfortran compilers.

### Configure gridgen

The mesh generation software [gridgen-c] is recommended to be installed through the Conda-Forge channel with Conda.
If [gridgen-c] is installed through Conda, the mesh generation code can locate the Dynamic Link Library **libgridgen.so** using the environment variable **\$CONDA_PREFIX**.
Otherwise it will look through the entries in **\$LD_LIBRARY_PATH** to look for **libgridgen.so**.
If neither of the above works, it will look for its own path **pyroms.__path__[0]** for **libgridgen.so**.

If [gridgen-c] is not installed through Conda, you need to configure **\$LD_LIBRARY_PATH** by adding these lines in your **.bashrc** file

```bash
# add entry to $LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH="/path/to/libgridgen.so:$LD_LIRBRAY_PATH"
```

## Uninstall

To remove pyROMS2 you can use the "pip uninstall" command

```bash
# uninstall pyROMS2
$ pip uninstall pyroms2
```

## Running pyROMS2

A jupyter-notebook project will be attached later to demonstrate the functionalities.


[ChuningWang/pyroms2]: https://github.com/ChuningWang/pyroms2
[ROMS]: https://www.myroms.org/
[ESMG/pyroms]: https://github.com/ESMG/pyroms
[python3]: https://github.com/ESMG/pyroms/tree/python3
[Cartopy]: https://scitools.org.uk/cartopy/docs/latest/
[Xarray]: https://xarray.pydata.org/en/stable/index.html
[cf-Xarray]: https://cf-xarray.readthedocs.io/en/latest/
[CF Metadata Convention]: https://cfconventions.org/
[Natural Earth]: https://www.naturalearthdata.com
[XROMS]: https://github.com/xoceanmodel/xroms
[Conda]: https://docs.conda.io/en/latest/
[Anaconda]: https://www.anaconda.com/
[Miniconda]: https://docs.conda.io/en/latest/miniconda.html
[conda-forge tips and tricks]: https://conda-forge.org/docs/user/tipsandtricks.html
[Conda-Forge]: https://conda-forge.org/
[xESMF]: https://xesmf.readthedocs.io/en/latest/
[numpy]: https://numpy.org/
[numba]: https://numba.pydata.org/
[scipy]: https://www.scipy.org/
[netcdf4]: https://unidata.github.io/netcdf4-python/netCDF4/index.html
[matplotlib]: https://matplotlib.org/
[shapely]: https://shapely.readthedocs.io/en/stable/manual.html
[pyproj]: https://pyproj4.github.io/pyproj/stable/
[cftime]: https://unidata.github.io/cftime/
[gridgen-c]: https://anaconda.org/conda-forge/gridgen
[pip]: https://pypi.org/project/pip/
[SCRIP]: https://github.com/SCRIP-Project/SCRIP
[Basemap]: https://basemaptutorial.readthedocs.io/en/latest/
[lpsolve55]: http://lpsolve.sourceforge.net/5.5/index.htm
[xgcm]: https://xgcm.readthedocs.io/en/latest/index.html
