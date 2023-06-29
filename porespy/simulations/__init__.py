r"""

Collection of functions for performing numerical simulations on images
######################################################################

This module contains routines for performing simulations directly on images.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    simulations.drainage
    simulations.tortuosity_fd

"""

from ._drainage import *
from ._dns import *
from ._ibip import *
from ._ibip_gpu import *
from ._ibip_w_gravity import *
