import time
import sys
from colorama import Fore


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


def display_timing_stats(desc, num_completed, total, start_time, end_time, file=sys.stdout, step_size=1,
                         colors=(Fore.BLUE, Fore.MAGENTA, Fore.YELLOW, Fore.CYAN)):
    item_index = num_completed - 1
    time_taken_str = eta_str = it_per_sec = ''
    time_delta = end_time - start_time
    if num_completed > 0 and time_delta > 0 and total is not None:
        time_taken_str = delta_time_string(time_delta)
        eta_str = delta_time_string(time_delta*(total-num_completed)/num_completed)
        it_per_sec = f'{(time_delta/num_completed):5.2f}s/it' if (time_delta > num_completed) \
            else f'{(num_completed/time_delta):5.2f}it/s'
    #
    if num_completed == 0 or (item_index * step_size) == 0 or (total is not None and num_completed == total):
        num_bars = int(num_completed*10.0/total) if total is not None else 0
        percentage = f'{num_completed*100.0/total:5.0f}%' if total is not None else '    '
        bar_string = f"{'#'*num_bars + ' '*(10-num_bars)}"
        file.write(f'\r{colors[0]}{desc}|'
                   f'{colors[1]}{bar_string}| '
                   f'{colors[2]}{percentage} {num_completed}/{total}| '
                   f'{colors[3]}[{time_taken_str}<{eta_str} {it_per_sec}]{Fore.RESET}')
        file.flush()
    #
