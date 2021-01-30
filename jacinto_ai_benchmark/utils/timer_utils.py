import time

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