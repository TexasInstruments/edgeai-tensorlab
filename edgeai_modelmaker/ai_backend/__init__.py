import os
import sys

from . import utils
from . import vision
from . import audio


def get_backend_module(backend_name):
    this_module = sys.modules[__name__]
    backend_module = getattr(this_module, backend_name)
    return backend_module
