import sys
import time
from colorama import Fore
from .timer_utils import delta_time_string

try:
    from tqdm.auto import tqdm
    has_tqdm = True
except:
    has_tqdm = False
#

__all__ = ['progress_step']


def progress_step(iterable, desc, desc_len=70, total=None, miniters=None, bar_format=None, file=sys.stdout,
                  leave=True, **kwargs):
    desc = desc[:desc_len] if (desc_len is not None and len(desc) > desc_len) else \
        desc + ' '*(desc_len-len(desc))
    if miniters is None:
        iter_length = len(iterable) if hasattr(iterable, '__len__') else total
        miniters = max(iter_length//100, 1) if iter_length is not None else 1
    #
    if bar_format is None:
        format_arg = (Fore.GREEN, desc_len, Fore.WHITE, Fore.YELLOW, Fore.CYAN, Fore.RESET)
        bar_format = '%s{desc:%s}|%s{percentage:4.0f}%%|%s{bar:10}|%s{r_bar}%s' % format_arg
    #
    return tqdm_step(iterable=iterable, desc=desc, total=total, miniters=miniters, bar_format=bar_format, file=file,
                leave=leave, **kwargs)


if has_tqdm:
    # use tqdm if its is available
    class tqdm_step(tqdm):
        def __init__(self, iterable, *args, **kwargs):
            super().__init__(iterable, *args, **kwargs)
            assert 'miniters' in kwargs, 'miniters must be used as keyword argument'
            self.step_size = kwargs['miniters']
            self.iter_index = 0
            self.steps_counted = 0

        def update(self, n=None):
            n = 1 if n is None else n
            steps_counted = self.iter_index//self.step_size
            if steps_counted > self.steps_counted:
                for _ in range(self.steps_counted, steps_counted):
                    super().update(self.step_size)
                #
                self.steps_counted = steps_counted
            elif hasattr(self, '__len__') and self.iter_index == (self.__len__()-1):
                super().update(self.iter_index % self.step_size)
            #
            self.iter_index += n

else:
    # simple implementation to be used only if tqdm is not available
    class tqdm_step:
        def __init__(self, iterable, desc, desc_len=70, miniters=1, total=None, file=None, **kwargs):
            super().__init__()
            self.step_size = miniters
            self.iterable = iterable
            self.desc = desc[:desc_len] if (desc_len is not None and len(desc) > desc_len) else \
                desc + ' '*(desc_len-len(desc))
            self.total = iterable.__len__() if hasattr(iterable, '__len__') else total
            self.file = file if file is not None else sys.stdout
            if hasattr(self.iterable, '__len__'):
                self.__len__ = self.iterable.__len__
            #

        def __iter__(self):
            start_time = time.time()
            eta_str = '<'
            for index, item in enumerate(self.iterable):
                num_completed = (index+1)
                if index > 0 and self.total is not None:
                    end_time = time.time()
                    delta_time = end_time - start_time
                    time_taken_str = delta_time_string(delta_time)
                    eta_str = delta_time_string(delta_time*(self.total-num_completed)/num_completed)
                    it_per_sec = (num_completed/delta_time)
                    eta_str = f"{time_taken_str}<{eta_str} {it_per_sec:5.2f}it/s"
                #
                if (index * self.step_size) == 0 or (self.total is not None and index == (self.total-1)):
                    num_bars = int(num_completed*10.0/self.total) if self.total is not None else 0
                    percentage = f'{num_completed*100.0/self.total:5.0f}%' if self.total is not None else '    '
                    bar_string = f"{'_'*num_bars + ' '*(10-num_bars)}"
                    self.file.write(f'\r{Fore.GREEN}{self.desc} {Fore.WHITE}{percentage} |'
                                    f'{Fore.YELLOW}{bar_string}| {Fore.CYAN}{index+1}/{self.total} | '
                                    f'[{eta_str}]{Fore.RESET}')
                    self.file.flush()
                #
                yield item

#