#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
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

import os
import sys
import errno


def is_url(download_entry):
    return isinstance(download_entry, str) and \
            (download_entry.startswith('http://') or download_entry.startswith('https://'))


def _absolute_path(relpath):
    if relpath is None:
        return relpath
    elif relpath.startswith('http://') or relpath.startswith('https://'):
        return relpath
    else:
        return os.path.abspath(os.path.expanduser(os.path.normpath(relpath)))


def absolute_path(relpath):
    if isinstance(relpath, (list,tuple)):
        return [_absolute_path(f) for f in relpath]
    else:
        return _absolute_path(relpath)


def remove_if_exists(path):
    try:
        os.remove(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def make_symlink(source, dest):
    if source is None or (not os.path.exists(source)):
        print(f'make_symlink failed - source: {source} is invalid')
        return
    #
    remove_if_exists(dest)
    if os.path.dirname(source) == os.path.dirname(dest):
        base_dir = os.path.dirname(source)
        cur_dir = os.getcwd()
        os.chdir(base_dir)
        os.symlink(os.path.basename(source), os.path.basename(dest))
        os.chdir(cur_dir)
    else:
        os.symlink(source, dest)
    #
