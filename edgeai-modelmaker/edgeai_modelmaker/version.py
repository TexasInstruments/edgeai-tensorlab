__version__ = '8.6'


def get_version():
    return __version__


def get_version_str(delimiter='_'):
    version_str = delimiter.join([f'{r:0>2}' for r in __version__.split('.')])
    return version_str
