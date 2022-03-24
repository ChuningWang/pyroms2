# pyROMS2

pyROMS2 is a collection of tools to process input and output files from the Regional Ocean Modeling System, [ROMS].
It was originally started by Rob Hetland as a googlecode project, then he morphed it into octant, also hosted on googlecode.
Frederic Castruccio then created a fork and renamed it back to pyROMS, and Kate Hedstrom is the current maintainer.

The offical pyROMS project is now hosted on GitHub.com in the [ESMG/pyroms] repository, default on the [python3] branch.
This branch is originally forked from [ESMG/pyroms], and then developed into a separate project hosted on [ChuningWang/pyroms2].
Many of the computation kernels are rewritten with modern Python packages, such as [Cartopy] and [Xarray].

## Installation

pyROMS2 is still a bit rough around the edges, particularly with regard to installation.
Recent development has been done in Python environments managed by [Conda].
However pyROMS2 itself cannot yet be installed with Conda.

If you are starting from scratch, we recommend that you install [Anaconda] or [Miniconda]
and create a Python 3 environment (as of March 2022, version 3.9 is your best bet).
You should also consider making conda-forge your default channel. See the [conda-forge tips and tricks] page.

If you don't want to use Conda, that's fine, but you will have to do more of the work yourself.

### Prerequisites

The following are required and are all available through [Conda-Forge].

  * Python >= 3.9 (Python 3.9 currently recommended for new environments)
  * [netcdf4]
  * [xESMF]
    * [scipy]
    * [Xarray]
      * [numpy]
      * [cftime]
      * ...
    * ...
  * [Cartopy]
    * [matplotlib]
    * ...
  * [xgcm]
  * [gridgen-c]
  * [pip] (if use pip for package installation)

The following packages are required in the offical branch, but removed in this branch

  * [SCRIP], which is the Spherical Coordinate Remapping and Interpolation Package.
[SCRIP] is no longer maintained by its development team.
The Python scrip code (a rather old version) is bundled in the offical
[pyROMS](https://github.com/ESMG/pyroms/tree/python3/pyroms/external/scrip) branch.

  * [Basemap], which is no longer supported by Python 3.9, thus is replaced by its successor [Cartopy].

  * [lpsolve55], which is the linear programmiing solver written in **C**.
[lpsolve55] is used when smoothing bathymetry with the
[LP_smoothing_rx0](https://github.com/ChuningWang/pyroms2/tree/main/pyroms/bathy_tools/lp_smoothing.py) function.
For some unknown reason it is very slow and crashes a lot when tested on a laptop.
For this reason, lpsolve55 is replaced by the [scipy] LP solver in the bathymetry smoothing algorithm.
[scipy] very likely also uses [lpsolve55] somewhere in its computation kernel, but is much better bundled in [scipy].

### Install using PIP

First, install miniconda. Then, we recommend creating a new environment and install the following dependence

```bash
# Create a conda environment for pyROMS2
$ conda create -n pyroms_env python=3.9
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
$ pip install -e .
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
[Conda]: https://docs.conda.io/en/latest/
[Anaconda]: https://www.anaconda.com/
[Miniconda]: https://docs.conda.io/en/latest/miniconda.html
[conda-forge tips and tricks]: https://conda-forge.org/docs/user/tipsandtricks.html
[Conda-Forge]: https://conda-forge.org/
[xESMF]: https://xesmf.readthedocs.io/en/latest/
[numpy]: https://numpy.org/
[scipy]: https://www.scipy.org/
[netcdf4]: https://unidata.github.io/netcdf4-python/netCDF4/index.html
[matplotlib]: https://matplotlib.org/
[cftime]: https://unidata.github.io/cftime/
[gridgen-c]: https://anaconda.org/conda-forge/gridgen
[pip]: https://pypi.org/project/pip/
[SCRIP]: https://github.com/SCRIP-Project/SCRIP
[Basemap]: https://basemaptutorial.readthedocs.io/en/latest/
[lpsolve55]: http://lpsolve.sourceforge.net/5.5/index.htm
[xgcm]: https://xgcm.readthedocs.io/en/latest/index.html
