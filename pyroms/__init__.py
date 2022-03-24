"""
Python toolkits for working with ROMS ocean models

Functionalities:

 - Curvilinear grid generation
 - Vertical coordinate generation
 - IC/BC generation
 - Data plotting/post processing
"""

from . import bathy_tools
from . import vgrid
from . import hgrid
from . import grid
from . import nesting
from . import io
from . import regrid
from . import sta_hgrid
from . import sta_grid
from . import utility
from . import xr

__authors__ = ['Frederic Castruccio (frederic@marine.rutgers.edu)',
               'Chuning Wang (wangchuning@sjtu.edu.cn)']
__version__ = '0.5.0'
