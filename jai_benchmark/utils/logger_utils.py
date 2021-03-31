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


class TeeLogger:
    def __init__(self, file_name):
        super().__init__()
        self.src_stream = sys.stdout
        self.dst_file = open(file_name, 'w') if file_name is not None else None
        sys.stdout = self

    def __del__(self):
        self.close()

    def write(self, message):
        if self.src_stream is not None:
            self.src_stream.write(message)
        #
        if self.dst_file is not None:
            self.dst_file.write(message)
        #
        self.flush()

    def flush(self):
        if self.src_stream is not None:
            self.src_stream.flush()
        #
        if self.dst_file is not None:
            self.dst_file.flush()
        #

    def isatty(self):
        return self.src_stream.isatty()

    def close(self):
        if self.src_stream is not None:
            sys.stdout = self.src_stream
            self.src_stream = None
        #
        if self.dst_file is not None:
            self.dst_file.close()
            self.dst_file = None
        #


class RedirectLogger:
    """
    Redirect the outputs of current process and subprocesses into a file
    """
    def __init__(self, file_name):
        super().__init__()
        self.src_stream = sys.stdout
        self.dst_file = open(file_name, 'w')
        # save a copy of stdout fd
        self.src_fd_orig = self.src_stream.fileno()
        self.src_fd_copy = os.dup(self.src_fd_orig)
        # connect file fd to stdout
        os.dup2(self.dst_file.fileno(), self.src_fd_orig)

    def __del__(self):
        self.close()

    def flush(self):
        self.src_stream.flush()
        if self.dst_file is not None:
            self.dst_file.flush()
        #

    def close(self):
        if self.dst_file is not None:
            self.dst_file.flush()
            self.src_stream.flush()
            # restore original stdout
            os.dup2(self.src_fd_copy, self.src_fd_orig)
            # cleanup
            os.close(self.src_fd_copy)
            self.dst_file.close()
            self.dst_file = None
        #


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
