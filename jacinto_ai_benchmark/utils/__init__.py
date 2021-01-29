import time
from .misc_utils import *
from .data_utils import *
from .model_utils import *
from .image_utils import *
from .logger_utils import *
from .metric_utils import *
from .parallel import Parallel

def download_model(url, root=None, filename=None, md5=None):
    root = os.path.abspath('./') if root is None else root
    if url.endswith('.link'):
        with open(url) as fp:
            url = fp.read().rstrip()
        #
    #
    if isinstance(url, str) and (url.startswith('http://') or url.startswith('https://')):
        fpath = download_and_extract_archive(url, root, filename, md5)
    else:
        fpath = url
    #
    return fpath


def timer(func, *args, repeats=1, **kwargs):
    start_time = time.time()
    for i in range(repeats):
        results = func(*args, **kwargs)
    #
    elapsed_time = (time.time() - start_time)/repeats
    return results, elapsed_time


# measures time spent on the current process.
# this is more accurate, but this will not measure time spent in os.system() calls
# so use timer() if you have os.system() calls.
def process_timer(func, *args, repeats=1, **kwargs):
    start_time = time.clock()
    for i in range(repeats):
        results = func(*args, **kwargs)
    #
    elapsed_process_time = (time.clock() - start_time)/repeats
    return results, elapsed_process_time