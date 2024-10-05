#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import sys
import logging

class BasicLogger:
    def __init__(self, filename=None, log_level=logging.INFO):
        assert log_level == logging.INFO, 'for now we support only INFO logging level'

        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

        if filename is not None:
            _file = logging.FileHandler(filename)
            self.logger.addHandler(_file)

        stdout = sys.stdout
        _stdo = logging.StreamHandler(stdout)
        _stdo.setLevel(log_level)
        self.logger.addHandler(_stdo)

        self.write = self.logger.info
        self.flush = stdout.flush
        self.fileno = stdout.fileno


class TeeLogger:
    def __init__(self, filename, log_level=logging.INFO, append=False, source_names=('stdout','stderr')):
        assert log_level == logging.INFO, 'for now we support only INFO logging level'
        mode = "a" if append else "w"
        self.source_names = source_names
        self.file = open(filename, mode)
        for source_name in self.source_names:
            source_stream = getattr(sys, source_name)
            setattr(self, source_name, source_stream)
            setattr(sys, source_name, self)
        #
        self.count = 0

    def __del__(self):
        self.close()

    def close(self):
        if self.file is None:
            return
        #
        for source_name in self.source_names:
            source_stream = getattr(self, source_name)
            setattr(sys, source_name, source_stream)
        #
        self.file.close()
        self.file = None

    def write(self, message):
        source_stream = getattr(self, self.source_names[0])
        source_stream.write(message)
        self.file.write(message)
        self.flush()

    def info(self, message):
        self.write(message)

    def debug(self, message):
        self.file.write(message)
        self.flush()

    def flush(self):
        for source_name in self.source_names:
            source_stream = getattr(self, source_name)
            source_stream.flush()
        #
        if self.file is None:
            return
        #
        self.file.flush()

    def fileno(self):
        return self.term.fileno()
