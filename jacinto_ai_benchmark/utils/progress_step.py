import sys
from tqdm.auto import tqdm
from colorama import Fore

__all__ = ['progress_step']


def progress_step(iterable, desc, desc_len=80, total=None, miniters=None, bar_format=None, file=sys.stdout,
                  leave=True, **kwargs):
    desc = desc[:desc_len] if (desc_len is not None and len(desc) > desc_len) else desc
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
