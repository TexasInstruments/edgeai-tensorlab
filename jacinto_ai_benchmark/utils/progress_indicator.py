
__all__ = ['progress_indicator']

import sys
import warnings
from tqdm.auto import tqdm
from colorama import Fore


def progress_indicator(iterable, desc):
    desc_len = 80
    desc = desc[:desc_len] if len(desc) > desc_len else desc
    miniters = max(len(iterable)//100, 1)
    colors = (Fore.GREEN, Fore.WHITE, Fore.YELLOW, Fore.CYAN, Fore.RESET)
    bar_format = '%s{desc:80}|%s{percentage:4.0f}%%|%s{bar:10}|%s{r_bar}%s' % colors
    return tqdm(iterable=iterable, desc=desc, bar_format=bar_format, file=sys.stdout,
                miniters=miniters, leave=True)
