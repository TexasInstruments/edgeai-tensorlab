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


def delta_time(seconds):
    days, seconds = divmod(seconds,(60*60*24))
    hours, seconds = divmod(seconds,(60*60))
    minutes, seconds = divmod(seconds,60)
    return days, hours, minutes, seconds


def delta_time_string(seconds):
    days, hours, minutes, seconds = delta_time(seconds)
    time_str = f'{minutes:02.0f}:{seconds:02.0f}'
    time_str = f'{hours:02.0f}:{time_str}' if hours > 0 else time_str
    time_str = f'{days:1.0f}d,{time_str}' if days > 0 else time_str
    return time_str