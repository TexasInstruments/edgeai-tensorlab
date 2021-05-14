# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import datetime
from colorama import Fore


def log_color(tag, title, message):
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if 'FATAL - {date}' in tag:
        msg = f'{Fore.RED}{tag}:{Fore.YELLOW}{date}: {title} - {Fore.RESET}{message}'
    elif 'ERROR' in tag:
        msg = f'{Fore.MAGENTA}{tag}:{Fore.YELLOW}{date}: {title} - {Fore.RESET}{message}'
    elif 'WARNING' in tag:
        msg = f'{Fore.CYAN}{tag}:{Fore.YELLOW}{date}: {title} - {Fore.RESET}{message}'
    elif 'INFO' in tag:
        msg = f'{Fore.BLUE}{tag}:{Fore.YELLOW}{date}: {title} - {Fore.RESET}{message}'
    elif 'SUCCESS' in tag:
        msg = f'{Fore.GREEN}{tag}:{Fore.YELLOW}{date}: {title} - {Fore.RESET}{message}'
    else:
        msg = f'{Fore.YELLOW}{tag}:{Fore.YELLOW}{date}: - {title} - {Fore.RESET}{message}'
    #
    return msg


class TeeLogger:
    def __init__(self, log_file, replace_stdout=False, append=False):
        super().__init__()
        self.replace_stdout = replace_stdout
        if isinstance(log_file, str):
            mode = 'a' if append else 'w'
            self.log_file = open(log_file, mode)
        else:
            self.log_file = log_file
        #
        if self.replace_stdout:
            self.log_stream = sys.stdout
            sys.stdout = self
        #

    def __del__(self):
        self.close()

    def write(self, message):
        sys.stdout.write(message)
        if self.log_file is not None:
            self.log_file.write(message)
        #
        self.flush()

    def flush(self):
        sys.stdout.flush()
        if self.log_file is not None:
            self.log_file.flush()
        #

    def isatty(self):
        return sys.stdout.isatty()

    def close(self):
        if self.replace_stdout:
            sys.stdout = self.log_stream
        #
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        #


##############################################################################
# RedirectLogger - Using the method suggested in:
# "Redirecting All Kinds of stdout in Python"
# https://dzone.com/articles/redirecting-all-kinds-stdout

import io
import sys
import ctypes

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')


class RedirectLogger():
    def __init__(self, dst_stream):
        self.original_stream = (sys.stdout, sys.stderr)
        self.dst_stream = dst_stream

    def __enter__(self):
        # this is the underlying src_fd
        self.original_fd = (sys.stdout.fileno(), sys.stderr.fileno())
        # take a backup of original_fd
        self.saved_fd = (os.dup(self.original_fd[0]), os.dup(self.original_fd[1]))
        # redirect original_fd to dst_fd
        self._redirect_fd(self.dst_stream.fileno())
        return self

    def __exit__(self, *args):
        # redirect original_fd to saved_fd
        self._redirect_fd(self.saved_fd)
        # is this needed?
        os.close(self.saved_fd[0])
        os.close(self.saved_fd[1])
        return True

    def _redirect_fd(self, to_fd):
        # flush the c_stdout
        libc.fflush(c_stdout)
        libc.fflush(c_stderr)
        # close stdout and its fd
        # sys.stdout.close()
        # sys.stderr.close()
        # make original_fd point to to_fd
        os.dup2(to_fd, self.original_fd[0])
        os.dup2(to_fd, self.original_fd[1])
        # make new sys.stdout
        sys.stdout = io.TextIOWrapper(os.fdopen(self.original_fd[0], 'wb'))
        sys.stderr = io.TextIOWrapper(os.fdopen(self.original_fd[1], 'wb'))
