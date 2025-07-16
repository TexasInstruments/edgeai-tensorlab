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
from setuptools import setup, Extension, find_packages


###############################################################################
def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm.tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    import hashlib
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def _get_redirect_url(url: str, max_hops: int = 10) -> str:
    import requests

    for hop in range(max_hops + 1):
        response = requests.get(url)

        if response.url == url or response.url is None:
            return url

        url = response.url
    else:
        raise RecursionError(f"ERROR: too many redirects: {max_hops + 1})")


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
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    #
    fpath = os.path.join(root, filename)

    # check if file is already present locally
    if (not force_download) and check_integrity(fpath, md5):
        print('INFO: using downloaded and verified file: ' + fpath)
        sys.stdout.flush()
        return fpath
    #

    print('INFO: downloading ' + url + ' to ' + fpath)
    sys.stdout.flush()

    os.makedirs(root, exist_ok=True)

    # expand redirect chain if needed
    if max_redirect_hops > 0:
        url = _get_redirect_url(url, max_hops=max_redirect_hops)
    #

    # download the file
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
        #
    #
    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
    #
    #print('done.')
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
    return _is_tar(from_path) or _is_targz(from_path) or \
           _is_gzip(from_path) or _is_zip(from_path) or _is_tgz(from_path)


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False,
                    verbose: bool = True, mode: Optional[str] = None):
    if verbose:
        print(f'INFO: extracting {from_path} to {to_path}')
        sys.stdout.flush()
    #
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        import tarfile
        mode = 'r' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='data')
    elif _is_targz(from_path) or _is_tgz(from_path):
        import tarfile
        mode = 'r:gz' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='data')
    elif _is_tarxz(from_path):
        import tarfile
        mode = 'r:xz' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='data')
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
    #
    if verbose:
        #print('done.')
        sys.stdout.flush()
    #
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
    url_files = url.split(' ')
    url = url_files[0]
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        # basename may contain http arguments after a ?. skip them
        filename = os.path.basename(url).split('?')[0]
    #
    if isinstance(url, str) and (url.startswith('http://') or url.startswith('https://')):
        fpath = download_url(url, download_root, filename, md5, force_download=force_download)
    else:
        fpath = url
    #
    if os.path.exists(fpath) and _is_archive(fpath):
        fpath = extract_archive(fpath, extract_root, remove_finished, mode=mode)
    #
    if len(url_files) > 1:
        fpath = os.path.join(fpath, url_files[1])
    #
    return fpath


###############################################################################
def download_arm_gcc(tidl_tools_package_path):
    print("INFO: installing gcc arm required for tvm...")
    GCC_ARM_AARCH64_NAME="arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"
    GCC_ARM_AARCH64_FILE=f"arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"
    GCC_ARM_AARCH64_PATH=f"https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/{GCC_ARM_AARCH64_FILE}"
    print(f"INFO: installing {tidl_tools_package_path}/{GCC_ARM_AARCH64_NAME}")
    if not os.path.exists(os.path.join(tidl_tools_package_path,GCC_ARM_AARCH64_NAME)):
        if not os.path.exists(os.path.join(tidl_tools_package_path,GCC_ARM_AARCH64_FILE)):
            # os.system(f"wget -P {tidl_tools_package_path} {GCC_ARM_AARCH64_PATH} --no-check-certificate")
            download_url(GCC_ARM_AARCH64_PATH, tidl_tools_package_path)
        #
        # os.system(f"tar xf {tidl_tools_package_path}/{GCC_ARM_AARCH64_FILE} -C ${TOOLS_BASE_PATH} > /dev/null")
        extract_archive(os.path.join(tidl_tools_package_path,GCC_ARM_AARCH64_FILE), tidl_tools_package_path)
    #


def download_tidl_tools(download_url, download_path, **tidl_version_dict):
    print("INFO: installing tidl_tools_package...")
    GCC_ARM_AARCH64_NAME="arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"
    cwd = os.getcwd()
    # download_path = os.path.join(tidl_tools_package_path, TARGET_SOC)
    download_tidl_tools_path = os.path.join(download_path, 'tidl_tools')
    shutil.rmtree(download_path, ignore_errors=True)
    os.makedirs(download_path, exist_ok=True)
    try:
        download_and_extract_archive(download_url, download_path, download_path)
        os.chdir(download_tidl_tools_path)
        os.symlink(os.path.join("..", "..", GCC_ARM_AARCH64_NAME), GCC_ARM_AARCH64_NAME)
        with open(os.path.join(download_tidl_tools_path, 'version.yaml'), "w") as fp:
            yaml.safe_dump(tidl_version_dict, fp)
        #
    except:
        print(f"ERROR: download_and_extract_archive: {download_url} - failed")
    #
    os.chdir(cwd)
    return None


###############################################################################
def download_tidl_tools_package_11_00(install_path, tools_version, tools_type):
    expected_tools_version=("11.0",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    tidl_tools_version_name=tools_version
    tidl_tools_release_label="r11.0"
    tidl_tools_release_id="11_00_08_00"
    c7x_firmware_version="11_00_00_00" #TODO - udpate this for 11.0
    c7x_firmware_version_possible_update=None #TODO - udpate this for 11.0
    print(f"INFO: you have chosen to install tidl_tools version:{tidl_tools_release_id} with default SDK firmware version set to:{c7x_firmware_version}")
    if c7x_firmware_version_possible_update:
        print(f"INFO: to leverage more features, set advanced_options:c7x_firmware_version while model compialtion and update firmware version in SDK to: {c7x_firmware_version_possible_update}")
        print(f"INFO: for more info, see version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md")
    #

    tidl_tools_package_path = install_path
    download_arm_gcc(tidl_tools_package_path)

    tidl_tools_type_suffix=("_gpu" if isinstance(tools_type,str) and "gpu" in tools_type else "")
    target_soc_download_urls = {
        "TDA4VM": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/TDA4VM",
        "AM68A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM68A",
        "AM69A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
        "AM67A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM67A",
        "AM62A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM62A", # no update for AM62A in 11.0
    }
    tidl_version_dict = dict(version=tidl_tools_version_name, release_label=tidl_tools_release_label,
                             release_id=tidl_tools_release_id, tools_type=tidl_tools_type_suffix,
                             c7x_firmware_version=c7x_firmware_version)
    for target_soc in target_soc_download_urls:
        download_url_base = target_soc_download_urls[target_soc]
        download_url = f"{download_url_base}/tidl_tools{tidl_tools_type_suffix}.tar.gz"
        download_path = os.path.join(tidl_tools_package_path, target_soc)
        download_tidl_tools(download_url, download_path, **tidl_version_dict, target_device=target_soc)
    #
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_11.0.txt'))
    return requirements_file


def download_tidl_tools_package_10_01(install_path, tools_version, tools_type):
    expected_tools_version=("10.1",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    tidl_tools_version_name=tools_version
    tidl_tools_release_label="r10.1"
    tidl_tools_release_id="10_01_04_01"
    c7x_firmware_version="10_01_03_00"
    c7x_firmware_version_possible_update="10_01_04_00"
    print(f"INFO: you have chosen to install tidl_tools version:{tidl_tools_release_id} with default SDK firmware version set to:{c7x_firmware_version}")
    print(f"INFO: to leverage more features, set advanced_options:c7x_firmware_version while model compialtion and update firmware version in SDK to: {c7x_firmware_version_possible_update}")
    print(f"INFO: for more info, see version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md")

    tidl_tools_package_path = install_path
    download_arm_gcc(tidl_tools_package_path)

    tidl_tools_type_suffix=("_gpu" if isinstance(tools_type,str) and "gpu" in tools_type else "")
    target_soc_download_urls = {
        "TDA4VM": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/TDA4VM",
        "AM68A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM68A",
        "AM69A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
        "AM67A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM67A",
        "AM62A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM62A",
    }
    tidl_version_dict = dict(version=tidl_tools_version_name, release_label=tidl_tools_release_label,
                             release_id=tidl_tools_release_id, tools_type=tidl_tools_type_suffix,
                             c7x_firmware_version=c7x_firmware_version)
    for target_soc in target_soc_download_urls:
        download_url_base = target_soc_download_urls[target_soc]
        download_url = f"{download_url_base}/tidl_tools{tidl_tools_type_suffix}.tar.gz"
        download_path = os.path.join(tidl_tools_package_path, target_soc)
        download_tidl_tools(download_url, download_path, **tidl_version_dict, target_device=target_soc)
    #
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_10.1.txt'))
    return requirements_file


def download_tidl_tools_package_10_00(install_path, tools_version, tools_type):
    expected_tools_version=("10.0",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    tidl_tools_version_name=tools_version
    tidl_tools_release_label="r10.0"
    tidl_tools_release_id="10_00_08_00"
    c7x_firmware_version=""
    print(f"INFO: you have chosen to install tidl_tools version:{tidl_tools_release_id} with default SDK firmware version:{c7x_firmware_version}")

    tidl_tools_package_path = install_path
    download_arm_gcc(tidl_tools_package_path)

    tidl_tools_type_suffix=("_gpu" if isinstance(tools_type,str) and "gpu" in tools_type else "")
    target_soc_download_urls = {
        "TDA4VM": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/TDA4VM",
        "AM68A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM68A",
        "AM69A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM69A",
        "AM67A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM67A",
        "AM62A": f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{tidl_tools_release_id}/TIDL_TOOLS/AM62A",
    }
    tidl_version_dict = dict(version=tidl_tools_version_name, release_label=tidl_tools_release_label,
                             release_id=tidl_tools_release_id, tools_type=tidl_tools_type_suffix,
                             c7x_firmware_version=c7x_firmware_version)
    for target_soc in target_soc_download_urls:
        download_url_base = target_soc_download_urls[target_soc]
        download_url = f"{download_url_base}/tidl_tools{tidl_tools_type_suffix}.tar.gz"
        download_path = os.path.join(tidl_tools_package_path, target_soc)
        download_tidl_tools(download_url, download_path, **tidl_version_dict, target_device=target_soc)
    #
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_10.0.txt'))
    return requirements_file


###############################################################################
down_tidl_tools_package_dict = {
    "11.0":   download_tidl_tools_package_11_00,
    "10.1":   download_tidl_tools_package_10_01,
    "10.0":   download_tidl_tools_package_10_00,
}


def setup_tidl_tools(install_path, tools_version, tools_type):
    assert tools_version in down_tidl_tools_package_dict.keys(), f"ERROR: unknown tools_version provided: {tools_version} at {__file__}"
    down_tidl_tools_package_func = down_tidl_tools_package_dict[tools_version]
    requirements_file = down_tidl_tools_package_func(install_path, tools_version, tools_type)
    os.system(f'pip install -r {requirements_file}')


###############################################################################
def main(args):
    readme_file = os.path.realpath(os.path.join(os.path.dirname(__file__), 'README.md'))
    with open(readme_file,  encoding="utf8") as readme:
        long_description = readme.read()

    setup(
        name='tidl_tools_package',
        version=args.tools_version,
        description='tidl_tools_package for edgeai-benchmark',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-benchmark',
        author='EdgeAI, TIDL & Analytics Algo Teams',
        author_email='edgeai-dev@list.ti.com',
        classifiers=[
            'Development Status :: 4 - Beta'
            'Programming Language :: Python :: 3.10'
        ],
        keywords = 'artifical intelligence, deep learning, image classification, object detection, semantic segmentation, quantization',
        python_requires='>=3.10',
        packages=find_packages(),
        include_package_data=True,
        project_urls={
            'Source': 'https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-benchmark',
            'Bug Reports': 'https://e2e.ti.com/support/processors-group/processors/tags/TIDL',
        },
    )


def main_download_tools(args):
    cur_dir = os.getcwd()
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_dir)

    try:
        import tidl_tools_package
        install_path = tidl_tools_package.__file__
    except:
        install_path = None
        raise RuntimeError(f"ERROR: FAILED install tidl tools with: {vars(args)}")

    if install_path:
        args.install_path = os.path.dirname(install_path)
        print(f'INFO: preparing to install tidl tools with: {vars(args)}')
        setup_tidl_tools(args.install_path, args.tools_version, args.tools_type)
    #
    os.chdir(cur_dir)


if __name__ == '__main__':
    TIDL_TOOLS_TYPE_DEFAULT = ""
    TIDL_TOOLS_VERSION_DEFAULT = "11.0"

    args = argparse.Namespace()
    args.tools_version = os.environ.get("TIDL_TOOLS_VERSION", TIDL_TOOLS_VERSION_DEFAULT)
    args.tools_type = os.environ.get("TIDL_TOOLS_TYPE", TIDL_TOOLS_TYPE_DEFAULT)
    args.setup_type = sys.argv[1]
    args.install_path = None
    main(args)

    if args.setup_type in ('develop', 'install'):
        main_download_tools(args)
    else:
        print("================================================ERROR======================================================")
        raise RuntimeError(f"ERROR: as of now this tidl_tools_package can be installed only develop in mode - but obtained the command:{args.setup_type}"
             f"\n    recommend to use one of these instead: "
             f"\n    pip install -e ./tools/"
             f"\n    python setup.py develop ./tools/")
        print("==========================================================================================================")