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
    if 'FATAL' in tag:
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
    def __init__(self, log_file, replace_stdout=False, append=False, buffering=-1):
        super().__init__()
        self.replace_stdout = replace_stdout
        if isinstance(log_file, str):
            mode = 'a' if append else 'w'
            # buffering=1 implies line buffering - i.e. file will be written to after each line
            # this may be slower than the default value, but will ensure more frequent file updates.
            self.log_file = open(log_file, mode, buffering=buffering)
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
# RedirectLogger - Inspired by: "Redirecting All Kinds of stdout in Python",
#                               https://dzone.com/articles/redirecting-all-kinds-stdout
# This method is also seen in: https://pypi.org/project/stream-redirect/

import io
import sys
import ctypes

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')


class RedirectStream():
    def __init__(self, dst_stream, stream_name='stdout'):
        self.stream_name = stream_name
        self.original_stream = getattr(sys, self.stream_name)
        self.dst_stream = dst_stream

    def begin(self):
        # this is the underlying src_fd
        self.original_fd = getattr(sys, self.stream_name).fileno()
        # take a backup of original_fd
        self.saved_fd = os.dup(self.original_fd)
        # flush
        self._flush()
        # redirect original_fd to dst_fd
        self._redirect_fd(self.dst_stream.fileno())
        return self

    def end(self, *args):
        # flush
        self._flush()
        # redirect original_fd to saved_fd
        self._redirect_fd(self.saved_fd)
        # is this needed?
        os.close(self.saved_fd)
        return True

    def _flush(self):
        # flush the c_stdout
        libc.fflush(c_stdout)
        libc.fflush(c_stderr)
        # close stdout and its fd
        getattr(sys, self.stream_name).flush()

    def _redirect_fd(self, to_fd):
        # close original stdout and its fd
        # observation: those close causes print issues in other process
        # created using 'spawn' (instead of 'fork')
        getattr(sys, self.stream_name).close()
        # make original_fd point to to_fd
        os.dup2(to_fd, self.original_fd)
        # make new sys.stdout
        new_stream = io.TextIOWrapper(os.fdopen(self.original_fd, 'wb'))
        setattr(sys, self.stream_name, new_stream)


class RedirectLogger():
    def __init__(self, dst_stream):
        self.stdout = RedirectStream(dst_stream, 'stdout')
        self.stderr = RedirectStream(dst_stream, 'stderr')

    def __enter__(self):
        self.stdout.begin()
        self.stderr.begin()
        return self

    def __exit__(self, *args):
        self.stdout.end()
        self.stderr.end()
        return True
