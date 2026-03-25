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

import sys
import os
from typing import Callable, Any, Optional
import importlib
import traceback
import shutil
import subprocess
import yaml
import argparse
import tqdm
import hashlib
import urllib.request
import urllib.error
import gzip


###############################################################################
TIDL_TOOLS_TYPE_DEFAULT = "cpu"
TIDL_TOOLS_VERSION_DEFAULT = "11.2"


TARGET_DEVICE_MAP = {
    'AM62'   : ['AM62','AM62X'],
    'AM62A'  : ['AM62A','AM62AX'],
    'J722S'  : ['J722S','AM67A','TDA4AEN'],
    'J721E'  : ['J721E','AM68PA','TDA4VM'],
    'J721S2' : ['J721S2','AM68A','TDA4VL','TDA4AL'],
    'J742S2' : ['J742S2','TDA4VP','TDA4AP'],
    'J784S4' : ['J784S4','AM69A','TDA4VH','TDA4AH'],
}


###############################################################################
def get_target_device(soc: str) -> str:
    """Get the target device from SOC name."""
    soc = soc.strip().upper()
    for key, val in TARGET_DEVICE_MAP.items():
        if soc in val:
            return key.strip().upper()
    return ''


def _create_symlink(src, dest):
    """Create a symbolic link from src to dest."""
    try:
        os.unlink(dest)
    except Exception:
        pass
    
    try:
        shutil.rmtree(dest, ignore_errors=True)
    except Exception:
        pass
    
    try:
        os.symlink(src, dest)
    except Exception:
        print(f'WARNING: symlink {src} -> {dest} - failed')


def _make_target_device_symlinks(tidl_tools_package_bin_path, target_device_map):
    """Create symlinks for target device aliases."""
    cur_dir = os.getcwd()
    os.chdir(tidl_tools_package_bin_path)
    for k, vlist in target_device_map.items():
        for v in vlist:
            if k != v and os.path.exists(k):
                _create_symlink(k, v)
    os.chdir(cur_dir)


###############################################################################
def gen_bar_updater() -> Callable[[int, int, int], None]:
    """Generate a progress bar updater for downloads."""
    pbar = tqdm.tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    """Calculate MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    """Verify MD5 checksum of a file."""
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    """Check if file exists and optionally verify its MD5 checksum."""
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def _get_redirect_url(url: str, max_hops: int = 10) -> str:
    """Follow URL redirects up to max_hops times."""
    import requests

    for hop in range(max_hops + 1):
        response = requests.get(url)

        if response.url == url or response.url is None:
            return url

        url = response.url
    else:
        raise RecursionError(f"ERROR: too many redirects: {max_hops + 1}")


def download_url(
    url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None,
    max_redirect_hops: int = 0, force_download: Optional[bool] = False):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed. eg: 3
        force_download (bool): whether to download even if the file exists
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    
    fpath = os.path.join(root, filename)

    # Check if file is already present locally
    if (not force_download) and check_integrity(fpath, md5):
        print('INFO: using downloaded and verified file: ' + fpath)
        sys.stdout.flush()
        return fpath

    print('INFO: downloading ' + url + ' to ' + fpath)
    sys.stdout.flush()

    os.makedirs(root, exist_ok=True)

    # Expand redirect chain if needed
    if max_redirect_hops > 0:
        url = _get_redirect_url(url, max_hops=max_redirect_hops)

    # Download the file
    try:
        urllib.request.urlretrieve(
            url, fpath,
            reporthook=gen_bar_updater()
        )
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('ERROR: failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + fpath)
            sys.stdout.flush()
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        else:
            raise e
    
    # Check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
    
    return fpath


def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")


def _is_archive(from_path):
    """Check if file is a supported archive format."""
    return (_is_tar(from_path) or _is_targz(from_path) or 
            _is_gzip(from_path) or _is_zip(from_path) or _is_tgz(from_path))


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False,
                    verbose: bool = True, mode: Optional[str] = None):
    """Extract archive file to specified path."""
    if verbose:
        print(f'INFO: extracting {from_path} to {to_path}')
        sys.stdout.flush()
    
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        import tarfile
        mode = 'r' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='tar')
    elif _is_targz(from_path) or _is_tgz(from_path):
        import tarfile
        mode = 'r:gz' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='tar')
    elif _is_tarxz(from_path):
        import tarfile
        mode = 'r:xz' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='tar')
    elif _is_gzip(from_path):
        mode = 'r' if mode is None else mode
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        import zipfile
        mode = 'r' if mode is None else mode
        with zipfile.ZipFile(from_path, mode) as z:
            z.extractall(to_path)
    else:
        raise ValueError("ERROR: extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)
    
    if verbose:
        sys.stdout.flush()
    
    return to_path


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
    mode: Optional[str] = None,
    force_download: Optional[bool] = False
):
    """Download and extract an archive file."""
    url_files = url.split(' ')
    url = url_files[0]
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        # Basename may contain http arguments after a ?. skip them
        filename = os.path.basename(url).split('?')[0]
    
    if isinstance(url, str) and (url.startswith('http://') or url.startswith('https://')):
        fpath = download_url(url, download_root, filename, md5, force_download=force_download)
    else:
        fpath = url
    
    if os.path.exists(fpath) and _is_archive(fpath):
        fpath = extract_archive(fpath, extract_root, remove_finished, mode=mode)
    
    if len(url_files) > 1:
        fpath = os.path.join(fpath, url_files[1])
    
    return fpath


###############################################################################
def download_arm_gcc(tidl_tools_package_bin_path):
    """Download and install ARM GCC toolchain required for TVM."""
    print("INFO: installing gcc arm required for tvm...")
    GCC_ARM_AARCH64_NAME = "arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"
    GCC_ARM_AARCH64_FILE = f"arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"
    GCC_ARM_AARCH64_PATH = f"https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/{GCC_ARM_AARCH64_FILE}"
    print(f"INFO: installing {tidl_tools_package_bin_path}/{GCC_ARM_AARCH64_NAME}")
    if not os.path.exists(os.path.join(tidl_tools_package_bin_path, GCC_ARM_AARCH64_NAME)):
        if not os.path.exists(os.path.join(tidl_tools_package_bin_path, GCC_ARM_AARCH64_FILE)):
            download_url(GCC_ARM_AARCH64_PATH, tidl_tools_package_bin_path)
        extract_archive(os.path.join(tidl_tools_package_bin_path, GCC_ARM_AARCH64_FILE), tidl_tools_package_bin_path)

def download_cgt_c7x(tidl_tools_package_bin_path, c7x_compiler_version):
    """Download and install C7x CGT compiler for TVM."""
    print("INFO: installing C7x CGT compiler for tvm...")
    CGT7X_NAME = f"ti-cgt-c7000_{c7x_compiler_version}"
    CGT7X_FILE = f"ti_cgt_c7000_{c7x_compiler_version}_linux-x64_installer.bin"
    CGT7X_PATH = f"https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-707zYe3Rik/{c7x_compiler_version}/{CGT7X_FILE}"
    print(f"INFO: installing {tidl_tools_package_bin_path}/{CGT7X_NAME}")
    if not os.path.exists(os.path.join(tidl_tools_package_bin_path, CGT7X_NAME)):
        if not os.path.exists(os.path.join(tidl_tools_package_bin_path, CGT7X_FILE)):
            download_url(CGT7X_PATH, tidl_tools_package_bin_path)
        
        # Make the installer executable
        installer_path = os.path.join(tidl_tools_package_bin_path, CGT7X_FILE)
        os.chmod(installer_path, 0o755)
        # Run the installer
        subprocess.run([installer_path, '--mode', 'unattended', '--prefix', tidl_tools_package_bin_path], check=True)


def download_tidl_tools(download_url, download_path, **tidl_version_dict):
    """Download and set up TIDL tools package."""
    print("INFO: installing tidl_tools_package...")
    GCC_ARM_AARCH64_NAME = "arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"
    CGT7X_NAME = f"ti-cgt-c7000_{tidl_version_dict.get('c7x_compiler_version', '5.0.0.LTS')}"

    cwd = os.getcwd()
    download_path = os.path.abspath(download_path)
    download_tidl_tools_path = os.path.join(download_path, 'tidl_tools')
    shutil.rmtree(download_path, ignore_errors=True)
    os.makedirs(download_path, exist_ok=True)
    try:
        download_and_extract_archive(download_url, download_path, download_path)
        os.chdir(download_tidl_tools_path)
        _create_symlink(os.path.join("..", "..", "..", GCC_ARM_AARCH64_NAME), GCC_ARM_AARCH64_NAME)
        _create_symlink(os.path.join("..", "..", GCC_ARM_AARCH64_NAME), GCC_ARM_AARCH64_NAME)
        _create_symlink(os.path.join("..", "..", CGT7X_NAME), CGT7X_NAME)
        with open(os.path.join(download_tidl_tools_path, 'version.yaml'), "w") as fp:
            yaml.safe_dump(tidl_version_dict, fp)
    except Exception as e:
        print(f"ERROR: download_and_extract_archive: {download_url} - failed - {e}")
    
    try:
        download_tidl_tools_path_until_rel, tidl_tools_name = os.path.split(download_tidl_tools_path)
        download_tidl_tools_path_until_dev, rel_name = os.path.split(download_tidl_tools_path_until_rel)
        download_tidl_tools_path_symlink_src = os.path.join(rel_name, tidl_tools_name)
        os.chdir(download_tidl_tools_path_until_dev)
        remove_link_stauts = os.unlink(tidl_tools_name) if os.path.islink(tidl_tools_name) else None
        remove_link_stauts = shutil.rmtree(tidl_tools_name) if os.path.exists(tidl_tools_name) else None
        os.symlink(download_tidl_tools_path_symlink_src, tidl_tools_name)
    except Exception as e:
        print(f"ERROR: symlink failed: {download_tidl_tools_path} to {download_tidl_tools_path_symlink_src} - {e}")
    
    os.chdir(cwd)
    return None


###############################################################################
def download_tidl_tools_package_11_02(install_path, tools_version, tools_type):
    """Download TIDL tools package version 11.02."""
    expected_tools_version = ("11.2",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    tidl_tools_version_name = tools_version
    tidl_tools_release_label = "r11.2"
    tidl_tools_release_id = "11_02_04_00"
    c7x_firmware_version = "11_02_04_00"
    c7x_compiler_version = "5.0.0.LTS"  # Needed for TVM (needs update based on release version)
    c7x_firmware_version_possible_update = None
    print(f"INFO: you have chosen to install tidl_tools version: {tidl_tools_release_id} with default SDK firmware version set to: {c7x_firmware_version}")
    if c7x_firmware_version_possible_update:
        print(f"INFO: to leverage more features, set advanced_options:c7x_firmware_version while model compialtion and update firmware version in SDK to: {c7x_firmware_version_possible_update}")
        print(f"INFO: for more info, see version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md")

    tidl_tools_package_bin_path = os.path.join(install_path, 'bin')
    download_arm_gcc(tidl_tools_package_bin_path)
    download_cgt_c7x(tidl_tools_package_bin_path, c7x_compiler_version)
    
    tidl_tools_type_suffix = ("_gpu" if isinstance(tools_type, str) and "gpu" in tools_type else "")
    target_soc_download_urls = {
        "AM62A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM62A",
        "J722S": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM67A",
        "J721E": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/TDA4VM",
        "J721S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM68A",
        "J742S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
        "J784S4": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
    }
    tidl_version_dict = dict(version=tidl_tools_version_name, release_label=tidl_tools_release_label,
                             release_id=tidl_tools_release_id, tools_type=tidl_tools_type_suffix,
                             c7x_firmware_version=c7x_firmware_version)
    for target_soc in target_soc_download_urls:
        download_url_base = target_soc_download_urls[target_soc]
        download_url = f"{download_url_base}/tidl_tools{tidl_tools_type_suffix}.tar.gz"
        download_path = os.path.join(tidl_tools_package_bin_path, target_soc, tidl_tools_release_id)
        download_tidl_tools(download_url, download_path, **tidl_version_dict, target_device=target_soc)
    
    _make_target_device_symlinks(tidl_tools_package_bin_path, TARGET_DEVICE_MAP)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_11.2.txt'))
    return requirements_file


###############################################################################
def download_tidl_tools_package_11_01(install_path, tools_version, tools_type):
    """Download TIDL tools package version 11.01."""
    expected_tools_version = ("11.1",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    tidl_tools_version_name = tools_version
    tidl_tools_release_label = "r11.1"
    tidl_tools_release_id = "11_01_06_00"
    c7x_firmware_version = "11_01_06_00"
    c7x_firmware_version_possible_update = None
    print(f"INFO: you have chosen to install tidl_tools version: {tidl_tools_release_id} with default SDK firmware version set to: {c7x_firmware_version}")
    if c7x_firmware_version_possible_update:
        print(f"INFO: to leverage more features, set advanced_options:c7x_firmware_version while model compialtion and update firmware version in SDK to: {c7x_firmware_version_possible_update}")
        print(f"INFO: for more info, see version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md")

    tidl_tools_package_bin_path = os.path.join(install_path, 'bin')
    download_arm_gcc(tidl_tools_package_bin_path)

    tidl_tools_type_suffix = ("_gpu" if isinstance(tools_type, str) and "gpu" in tools_type else "")
    target_soc_download_urls = {
        "AM62A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM62A",
        "J722S": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM67A",
        "J721E": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/TDA4VM",
        "J721S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM68A",
        "J742S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
        "J784S4": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
    }
    tidl_version_dict = dict(version=tidl_tools_version_name, release_label=tidl_tools_release_label,
                             release_id=tidl_tools_release_id, tools_type=tidl_tools_type_suffix,
                             c7x_firmware_version=c7x_firmware_version, c7x_compiler_version=None)
    for target_soc in target_soc_download_urls:
        download_url_base = target_soc_download_urls[target_soc]
        download_url = f"{download_url_base}/tidl_tools{tidl_tools_type_suffix}.tar.gz"
        download_path = os.path.join(tidl_tools_package_bin_path, target_soc, tidl_tools_release_id)
        download_tidl_tools(download_url, download_path, **tidl_version_dict, target_device=target_soc)
    
    _make_target_device_symlinks(tidl_tools_package_bin_path, TARGET_DEVICE_MAP)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_11.1.txt'))
    return requirements_file


###############################################################################
def download_tidl_tools_package_11_00(install_path, tools_version, tools_type):
    """Download TIDL tools package version 11.00."""
    expected_tools_version = ("11.0",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    tidl_tools_version_name = tools_version
    tidl_tools_release_label = "r11.0"
    tidl_tools_release_id = "11_00_06_00" #using 11_00_06_00 to match with SDK, but 11_00_08_00 is also available
    c7x_firmware_version = "11_00_00_00"  # TODO - update this for 11.0
    c7x_firmware_version_possible_update = None  # TODO - update this for 11.0
    print(f"INFO: you have chosen to install tidl_tools version: {tidl_tools_release_id} with default SDK firmware version set to: {c7x_firmware_version}")
    if c7x_firmware_version_possible_update:
        print(f"INFO: to leverage more features, set advanced_options:c7x_firmware_version while model compialtion and update firmware version in SDK to: {c7x_firmware_version_possible_update}")
        print(f"INFO: for more info, see version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md")

    tidl_tools_package_bin_path = os.path.join(install_path, 'bin')
    download_arm_gcc(tidl_tools_package_bin_path)

    tidl_tools_type_suffix = ("_gpu" if isinstance(tools_type, str) and "gpu" in tools_type else "")
    target_soc_download_urls = {
        "AM62A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM62A", # no update for AM62A in 11.0
        "AM67A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM67A",
        "J721E": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/TDA4VM",
        "J721S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM68A",
        "J742S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
        "J784S4": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
    }
    tidl_version_dict = dict(version=tidl_tools_version_name, release_label=tidl_tools_release_label,
                             release_id=tidl_tools_release_id, tools_type=tidl_tools_type_suffix,
                             c7x_firmware_version=c7x_firmware_version)
    for target_soc in target_soc_download_urls:
        download_url_base = target_soc_download_urls[target_soc]
        download_url = f"{download_url_base}/tidl_tools{tidl_tools_type_suffix}.tar.gz"
        download_path = os.path.join(tidl_tools_package_bin_path, target_soc, tidl_tools_release_id)
        download_tidl_tools(download_url, download_path, **tidl_version_dict, target_device=target_soc)
    
    _make_target_device_symlinks(tidl_tools_package_bin_path, TARGET_DEVICE_MAP)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_11.0.txt'))
    return requirements_file


###############################################################################
def download_tidl_tools_package_10_01(install_path, tools_version, tools_type):
    """Download TIDL tools package version 10.01."""
    expected_tools_version = ("10.1",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    tidl_tools_version_name = tools_version
    tidl_tools_release_label = "r10.1"
    tidl_tools_release_id = "10_01_04_01"
    c7x_firmware_version = "10_01_03_00"
    c7x_firmware_version_possible_update = "10_01_04_00"
    print(f"INFO: you have chosen to install tidl_tools version: {tidl_tools_release_id} with default SDK firmware version set to: {c7x_firmware_version}")
    print(f"INFO: to leverage more features, set advanced_options:c7x_firmware_version while model compialtion and update firmware version in SDK to: {c7x_firmware_version_possible_update}")
    print(f"INFO: for more info, see version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md")

    tidl_tools_package_bin_path = os.path.join(install_path, 'bin')
    download_arm_gcc(tidl_tools_package_bin_path)

    tidl_tools_type_suffix = ("_gpu" if isinstance(tools_type, str) and "gpu" in tools_type else "")
    target_soc_download_urls = {
        "AM62A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM62A",
        "J722S": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM67A",
        "J721E": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/TDA4VM",
        "J721S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM68A",
        "J742S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
        "J784S4": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
    }
    tidl_version_dict = dict(version=tidl_tools_version_name, release_label=tidl_tools_release_label,
                             release_id=tidl_tools_release_id, tools_type=tidl_tools_type_suffix,
                             c7x_firmware_version=c7x_firmware_version)
    for target_soc in target_soc_download_urls:
        download_url_base = target_soc_download_urls[target_soc]
        download_url = f"{download_url_base}/tidl_tools{tidl_tools_type_suffix}.tar.gz"
        download_path = os.path.join(tidl_tools_package_bin_path, target_soc, tidl_tools_release_id)
        download_tidl_tools(download_url, download_path, **tidl_version_dict, target_device=target_soc)
    
    _make_target_device_symlinks(tidl_tools_package_bin_path, TARGET_DEVICE_MAP)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_10.1.txt'))
    return requirements_file


def download_tidl_tools_package_10_00(install_path, tools_version, tools_type):
    """Download TIDL tools package version 10.00."""
    expected_tools_version = ("10.0",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    tidl_tools_version_name = tools_version
    tidl_tools_release_label = "r10.0"
    tidl_tools_release_id = "10_00_08_00"
    c7x_firmware_version = ""
    print(f"INFO: you have chosen to install tidl_tools version: {tidl_tools_release_id} with default SDK firmware version: {c7x_firmware_version}")

    tidl_tools_package_bin_path = os.path.join(install_path, 'bin')
    download_arm_gcc(tidl_tools_package_bin_path)

    tidl_tools_type_suffix = ("_gpu" if isinstance(tools_type, str) and "gpu" in tools_type else "")
    target_soc_download_urls = {
        "AM62A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM62A",
        "J722S": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM67A",
        "J721E": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/TDA4VM",
        "J721S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM68A",
        "J742S2": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
        "J784S4": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
    }
    tidl_version_dict = dict(version=tidl_tools_version_name, release_label=tidl_tools_release_label,
                             release_id=tidl_tools_release_id, tools_type=tidl_tools_type_suffix,
                             c7x_firmware_version=c7x_firmware_version)
    for target_soc in target_soc_download_urls:
        download_url_base = target_soc_download_urls[target_soc]
        download_url = f"{download_url_base}/tidl_tools{tidl_tools_type_suffix}.tar.gz"
        download_path = os.path.join(tidl_tools_package_bin_path, target_soc, tidl_tools_release_id)
        download_tidl_tools(download_url, download_path, **tidl_version_dict, target_device=target_soc)
    
    _make_target_device_symlinks(tidl_tools_package_bin_path, TARGET_DEVICE_MAP)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_10.0.txt'))
    return requirements_file


###############################################################################
down_tidl_tools_package_dict = {
    "11.2": download_tidl_tools_package_11_02,
    "11.1": download_tidl_tools_package_11_01,
    "11.0": download_tidl_tools_package_11_00,
    "10.1": download_tidl_tools_package_10_01,
    "10.0": download_tidl_tools_package_10_00,
}


def setup_tidl_tools(install_path, tools_version, tools_type):
    """Set up TIDL tools with the specified version and type."""
    assert tools_version in down_tidl_tools_package_dict.keys(), f"ERROR: unknown tools_version provided: {tools_version} at {__file__}"
    down_tidl_tools_package_func = down_tidl_tools_package_dict[tools_version]

    try:
        requirements_file = down_tidl_tools_package_func(install_path, tools_version, tools_type)
    except Exception as e:
        print(f"ERROR: download_tidl_tools_package failed for version: {tools_version} - {e}")
        traceback.print_exc()
    
    # Use sys.executable to ensure we use the same Python interpreter
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to install requirements from {requirements_file}: {e}")
        print(f"Please manually run: {sys.executable} -m pip3 install -r {requirements_file}")
        traceback.print_exc()


###############################################################################
# this function is the entrypoint for download_tidlrunner_tools as specified in pyproject.toml
def install_package(*install_args, install_cmd="install"):
    """Install osrt_model_tools package."""
    
    package_name = install_args[0].split('@')[0].split('==')[0]
    # Check if package is already installed
    if install_cmd == "install" and importlib.util.find_spec(package_name) is not None:
        print(f"INFO: {package_name} is already installed")
        return True
    try:
        print(f"INFO: {install_cmd} {package_name}")
        install_options = [str(arg) for arg in install_args]
        install_cmd_list = ["python", "-m", "pip", install_cmd, "--no-input", "--yes"] + install_options
        print(f"INFO: {install_cmd} {package_name} using:", " ".join(install_cmd_list))
        result = subprocess.run(install_cmd_list, check=True, capture_output=True, text=True)
        
        print(f"SUCCESS: {package_name} installed successfully")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to {install_cmd} {package_name}")
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
    """Uninstall a package."""
    install_package(*install_args, install_cmd=install_cmd)


def cleanup_tidl_tools_old(install_path, target_device_map):
    for k, target_device_list in target_device_map.items():
        for target_device in target_device_list:
            target_device_path_old = os.path.join(install_path, target_device)
            if os.path.islink(target_device_path_old):
                os.unlink(target_device_path_old)
            else:
                shutil.rmtree(target_device_path_old, ignore_errors=True)


###############################################################################
# this function is the entrypoint for download_tidl_tools as specified in pyproject.toml
def download():
    """Download and set up TIDL tools."""
    uninstall_package("onnxruntime-tidl")
    uninstall_package("onnxruntime")
    uninstall_package("onnx")
    uninstall_package("protobuf")
    uninstall_package("osrt-model-tools")

    install_path = os.path.dirname(os.path.realpath(__file__))
    tools_version = os.environ.get("TIDL_TOOLS_VERSION", TIDL_TOOLS_VERSION_DEFAULT)
    tools_type = os.environ.get("TIDL_TOOLS_TYPE", TIDL_TOOLS_TYPE_DEFAULT)

    print(f"INFO: cleaning up any old TIDL tools installation at {install_path}...")
    cleanup_tidl_tools_old(install_path, TARGET_DEVICE_MAP)

    print(f"INFO: running setup with TIDL_TOOLS_VERSION={tools_version} TIDL_TOOLS_TYPE={tools_type}")
    print(f"INFO: tidl-tools will be installed in {os.path.join(install_path, 'bin')}")
    setup_tidl_tools(install_path, tools_version, tools_type)


def main():
    download()


if __name__ == '__main__':
    main()
