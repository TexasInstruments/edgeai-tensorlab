import os

def set_pythonpath(additional_paths=None):
    '''set PYTHONPATH'''
    PYTHONPATH = os.environ.get('PYTHONPATH', None)
    if PYTHONPATH is None:
        PYTHONPATH = ''
    #
    if additional_paths is not None:
        PYTHONPATH = ':'+additional_paths+PYTHONPATH
    #
    if not PYTHONPATH.startswith(':'):
        PYTHONPATH = ':'+PYTHONPATH
    #
    os.environ['PYTHONPATH'] = PYTHONPATH

