import sys
import importlib
import os

def import_file_or_folder(folder_or_file_name, package_name=None, force_import=False):
    if folder_or_file_name.endswith(os.sep):
        folder_or_file_name = folder_or_file_name[:-1]
    #
    if folder_or_file_name.endswith('.py'):
        folder_or_file_name = folder_or_file_name[:-3]
    #
    parent_folder = os.path.dirname(folder_or_file_name)
    basename = os.path.basename(folder_or_file_name)
    if force_import:
        sys.modules.pop(basename, None)
    #
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, package_name or __name__)
    sys.path.pop(0)
    return imported_module


    

