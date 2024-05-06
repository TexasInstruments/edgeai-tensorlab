import os
import sys
import importlib

__all__ = ['import_file', 'import_folder']

def import_package(folder_name):
    if folder_name.endswith(os.sep):
        folder_name = folder_name[:-1]
    #
    parent_folder = os.path.dirname(folder_name)
    basename = os.path.basename(folder_name)
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, __name__)
    sys.path.pop(0)
    return imported_module

import_folder = import_package

def import_file(file_name):
    module_name = file_name.rsplit('.', 1)[0]
    return import_package(module_name)
