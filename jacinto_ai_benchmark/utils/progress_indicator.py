import sys
from tqdm.auto import tqdm
from colorama import Fore

__all__ = ['progress_indicator']


def progress_indicator(iterable, desc, desc_len=80):
    desc = desc[:desc_len] if (desc_len is not None and len(desc) > desc_len) else desc
    miniters = max(len(iterable)//100, 1)
    format_arg = (Fore.GREEN, desc_len, Fore.WHITE, Fore.YELLOW, Fore.CYAN, Fore.RESET)
    bar_format = '%s{desc:%s}|%s{percentage:4.0f}%%|%s{bar:10}|%s{r_bar}%s' % format_arg
    return tqdm(iterable=iterable, desc=desc, bar_format=bar_format, file=sys.stdout,
                miniters=miniters, position=0, leave=True)
