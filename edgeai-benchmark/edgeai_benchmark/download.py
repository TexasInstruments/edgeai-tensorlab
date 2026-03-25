# Copyright (c) 2018-2025, Texas Instruments
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
import subprocess
import importlib.util


###############################################################################
# this function is the entrypoint for download_tidlrunner_tools as specified in pyproject.toml
def install_package(*install_args, install_cmd="install"):
    """Install osrt_model_tools package."""
    
    _package_name = install_args[0].split('@')[0].split('==')[0]
    # Check if package is already installed
    if importlib.util.find_spec(_package_name) is not None:
        print(f"INFO: {_package_name} is already installed")
        return True
    try:
        print(f"INFO: Installing {_package_name}")
        install_options = [str(arg) for arg in install_args]
        install_cmd_list = ["python3", "-m", "pip", install_cmd, "--no-input"] + install_options
        print(f"INFO: installing {_package_name} using:", " ".join(install_cmd_list))
        result = subprocess.run(install_cmd_list, check=True, capture_output=True, text=True)
        
        print(f"SUCCESS: {_package_name} installed successfully")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install {_package_name}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error during installation: {e}")
        return False


def uninstall_package(*install_args, install_cmd="uninstall"):
    install_package(*install_args, install_cmd=install_cmd)


def main():
    install_package("onnx-graphsurgeon==0.3.26", "--extra-index-url", "https://pypi.ngc.nvidia.com")
    # install_package("onnx-graphsurgeon==0.5.8", "--extra-index-url", "https://pypi.ngc.nvidia.com")


if __name__ == '__main__':
    main()