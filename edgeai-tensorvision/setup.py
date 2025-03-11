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
import subprocess
from setuptools import setup, Extension, find_packages


def git_hash():
    git_path = './' if os.path.exists('.git') else ('../' if os.path.exists('../.git') else None)
    if git_path:
        hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return hash[:7] if (hash is not None) else None
    else:
        return None


def get_version():
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'version.txt')
    with open(version_file) as f:
        __version__ = f.readline().strip()
    hash = git_hash()
    version_str = __version__ + '+' + hash.strip().decode('ascii') if (hash is not None) else __version__
    return version_str


if __name__ == '__main__':
    version_str = get_version()

    long_description = ''
    with open('README.md',  encoding="utf8") as readme:
        long_description = readme.read()

    setup(
        name='edgeai_tensorvision',
        version=get_version(),
        description='Edge AI Vision Models',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/TexasInstruments/edgeai-tensorlab',
        author='Texas Instruments - Edge AI / Analytics Algo Teams',
        author_email='edgeai-tensorlab@list.ti.com',
        classifiers=[
            'Development Status :: 4 - Beta'
            'Programming Language :: Python :: 3.10'
        ],
        keywords = 'artifical intelligence, deep learning, image classification, object detection, semantic segmentation, quantization',
        python_requires='>=3.10',
        packages=find_packages(),
        include_package_data=True,
        install_requires=[],
        project_urls={
            'Source': 'https://github.com/TexasInstruments/edgeai-tensorlab',
            'Bug Reports': 'https://github.com/TexasInstruments/edgeai-tensorlab/issues',
        },
    )

