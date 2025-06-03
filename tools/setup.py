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
from setuptools import setup, Extension, find_packages
import yaml
import argparse
import tqdm


def git_hash():
    git_path = './' if os.path.exists('.git') else ('../' if os.path.exists('../.git') else None)
    if git_path:
        hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return hash[:7] if (hash is not None) else None
    else:
        return None


def import_file_folder(file_or_folder_name):
    if file_or_folder_name.endswith(os.sep):
        file_or_folder_name = file_or_folder_name[:-1]
    #
    parent_folder = os.path.dirname(file_or_folder_name)
    basename = os.path.splitext(os.path.basename(file_or_folder_name))[0]
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, __name__)
    sys.path.pop(0)
    return imported_module


def get_version():
    version_file = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'edgeai_benchmark', 'version.py'))
    print(f"version_file={version_file}")
    version = import_file_folder(version_file)
    hash = git_hash()
    version_str = version.__version__ + '+' + hash.strip().decode('ascii') if (hash is not None) else version.__version__
    return version_str


def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm.tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
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


def download_tidl_tools(TIDL_TOOLS_VERSION_NAME, TIDL_TOOLS_RELEASE_LABEL, TIDL_TOOLS_RELEASE_ID, TARGET_SOCS, TIDL_TOOLS_TYPE_SUFFIX, C7X_FIRMWARE_VERSION, DOWNLOAD_URLS=None):
    tidl_tools_package_path = os.path.join(os.path.dirname(__file__), 'tidl_tools_package')

    print("INFO: installing gcc arm required for tvm...")
    GCC_ARM_AARCH64_NAME="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
    GCC_ARM_AARCH64_FILE=f"{GCC_ARM_AARCH64_NAME}.tar.xz"
    GCC_ARM_AARCH64_PATH=f"https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/{GCC_ARM_AARCH64_FILE}"
    print(f"INFO: installing {tidl_tools_package_path}/{GCC_ARM_AARCH64_NAME}")
    if not os.path.exists(os.path.join(tidl_tools_package_path,GCC_ARM_AARCH64_NAME)):
        if not os.path.exists(os.path.join(tidl_tools_package_path,GCC_ARM_AARCH64_FILE)):
            # os.system(f"wget -P {tidl_tools_package_path} {GCC_ARM_AARCH64_PATH} --no-check-certificate")
            download_url(GCC_ARM_AARCH64_PATH, tidl_tools_package_path)
        #
        # os.system(f"tar xf {tidl_tools_package_path}/{GCC_ARM_AARCH64_FILE} -C ${TOOLS_BASE_PATH} > /dev/null")
        extract_archive(os.path.join(tidl_tools_package_path,GCC_ARM_AARCH64_FILE), tidl_tools_package_path)
    #

    print("INFO: installing tidl_tools_package...")
    cwd = os.getcwd()
    target_soc_version_dict = {}
    for TARGET_SOC in TARGET_SOCS:
        if isinstance(DOWNLOAD_URLS, dict):
            url_name = DOWNLOAD_URLS[TARGET_SOC]
        else:
            url_name = f"https://software-dl.ti.com/jacinto7/esd/tidl-tools/{TIDL_TOOLS_RELEASE_ID}/TIDL_TOOLS/{TARGET_SOC}/tidl_tools{TIDL_TOOLS_TYPE_SUFFIX}.tar.gz"
        #
        install_target_soc_path = os.path.join(tidl_tools_package_path, TARGET_SOC)
        install_path = os.path.join(install_target_soc_path, 'tidl_tools')

        shutil.rmtree(install_target_soc_path, ignore_errors=True)
        os.makedirs(install_target_soc_path, exist_ok=True)

        try:
            download_and_extract_archive(url_name, install_target_soc_path, install_target_soc_path)
            os.chdir(install_path)
            os.symlink(os.path.join("..", "..", GCC_ARM_AARCH64_NAME), GCC_ARM_AARCH64_NAME)

            version_dict = {
                "version": TIDL_TOOLS_VERSION_NAME,
                "release_label": TIDL_TOOLS_RELEASE_LABEL,
                "target_device": TARGET_SOC,
                "release_id": TIDL_TOOLS_RELEASE_ID,
                "c7x_firmware_version": C7X_FIRMWARE_VERSION
            }
            with open(os.path.join(install_path, 'version.yaml'), "w") as fp:
                yaml.safe_dump(version_dict, fp)
            #
            target_soc_version_dict.update({TARGET_SOC: version_dict})
            #print(target_soc_version_dict)
            os.chdir(cwd)
        except:
            print(f"ERROR: download_and_extract_archive: {url_name} - failed")
            os.chdir(cwd)
        #

    os.chdir(cwd)
    return None


def download_tidl_tools_package_11_00_06_00(tools_version, tools_type):
    expected_tools_version=("11.0",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    TIDL_TOOLS_VERSION_NAME=tools_version
    TIDL_TOOLS_RELEASE_LABEL="r11.0"
    TIDL_TOOLS_RELEASE_ID="11_00_06_00"
    C7X_FIRMWARE_VERSION="11_00_00_00"
    C7X_FIRMWARE_VERSION_POSSIBLE_UPDATE=None #TODO - udpate this
    TARGET_SOCS=("TDA4VM", "AM68A", "AM69A", "AM67A", "AM62A")
    TIDL_TOOLS_TYPE_SUFFIX=tools_type
    print(f"INFO: you have chosen to install tidl_tools version:{TIDL_TOOLS_RELEASE_ID} with default SDK firmware version set to:{C7X_FIRMWARE_VERSION}")
    if C7X_FIRMWARE_VERSION_POSSIBLE_UPDATE:
        print(f"INFO: to leverage more features, set advanced_options:c7x_firmware_version while model compialtion and update firmware version in SDK to: {C7X_FIRMWARE_VERSION_POSSIBLE_UPDATE}")
    #
    print(f"INFO: for more info, see version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md")
    download_tidl_tools(TIDL_TOOLS_VERSION_NAME, TIDL_TOOLS_RELEASE_LABEL, TIDL_TOOLS_RELEASE_ID, TARGET_SOCS, TIDL_TOOLS_TYPE_SUFFIX, C7X_FIRMWARE_VERSION)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_11.0.txt'))
    return requirements_file


def download_tidl_tools_package_10_01_04_01(tools_version, tools_type):
    expected_tools_version=("10.1",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    TIDL_TOOLS_VERSION_NAME=tools_version
    TIDL_TOOLS_RELEASE_LABEL="r10.1"
    TIDL_TOOLS_RELEASE_ID="10_01_04_01"
    C7X_FIRMWARE_VERSION="10_01_03_00"
    C7X_FIRMWARE_VERSION_POSSIBLE_UPDATE="10_01_04_00"
    TARGET_SOCS=("TDA4VM", "AM68A", "AM69A", "AM62A", "AM67A")
    TIDL_TOOLS_TYPE_SUFFIX=tools_type
    print(f"INFO: you have chosen to install tidl_tools version:{TIDL_TOOLS_RELEASE_ID} with default SDK firmware version set to:{C7X_FIRMWARE_VERSION}")
    print(f"INFO: to leverage more features, set advanced_options:c7x_firmware_version while model compialtion and update firmware version in SDK to: {C7X_FIRMWARE_VERSION_POSSIBLE_UPDATE}")
    print(f"INFO: for more info, see version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md")
    download_tidl_tools(TIDL_TOOLS_VERSION_NAME, TIDL_TOOLS_RELEASE_LABEL, TIDL_TOOLS_RELEASE_ID, TARGET_SOCS, TIDL_TOOLS_TYPE_SUFFIX, C7X_FIRMWARE_VERSION)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_10.1.txt'))
    return requirements_file


def download_tidl_tools_package_10_00_08_00(tools_version, tools_type):
    expected_tools_version=("10.0",)
    assert tools_version in expected_tools_version, f"ERROR: incorrect tools_version passed:{tools_version} - expected:{expected_tools_version}"
    TIDL_TOOLS_VERSION_NAME=tools_version
    TIDL_TOOLS_RELEASE_LABEL="r10.0"
    TIDL_TOOLS_RELEASE_ID="10_00_08_00"
    C7X_FIRMWARE_VERSION=""
    TARGET_SOCS=("TDA4VM", "AM68A", "AM69A", "AM62A", "AM67A")
    TIDL_TOOLS_TYPE_SUFFIX=tools_type
    print(f"INFO: you have chosen to install tidl_tools version:{TIDL_TOOLS_RELEASE_ID} with default SDK firmware version:{C7X_FIRMWARE_VERSION}")
    download_tidl_tools(TIDL_TOOLS_VERSION_NAME, TIDL_TOOLS_RELEASE_LABEL, TIDL_TOOLS_RELEASE_ID, TARGET_SOCS, TIDL_TOOLS_TYPE_SUFFIX, C7X_FIRMWARE_VERSION)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), f'requirements/requirements_10.0.txt'))
    return requirements_file


down_tidl_tools_package_dict = {
    "11.0":   download_tidl_tools_package_11_00_06_00,
    "10.1":   download_tidl_tools_package_10_01_04_01,
    "10.0":   download_tidl_tools_package_10_00_08_00,
}


def main(tools_version, tools_type):
    assert tools_version in down_tidl_tools_package_dict.keys(), f"unknown tools_version provided: {tools_version} at {__file__}"

    requirements_file = down_tidl_tools_package_dict[tools_version](tools_version, tools_type)
    # os.system(f'pip install -r {requirements_file}')

    readme_file = os.path.realpath(os.path.join(os.path.dirname(__file__), 'README.md'))
    with open(readme_file,  encoding="utf8") as readme:
        long_description = readme.read()

    try:
        setup(
            name='tidl_tools_package',
            version=tools_version,
            description='tidl-tools-package for model compilation',
            long_description=long_description,
            long_description_content_type='text/markdown',
            url='https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-benchmark/tools',
            author='EdgeAI, TIDL & Analytics Algo Teams',
            author_email='edgeai-devkit@list.ti.com',
            classifiers=[
                'Development Status :: 4 - Beta'
                'Programming Language :: Python :: 3.10'
            ],
            keywords = 'artifical intelligence, deep learning, image classification, object detection, semantic segmentation, quantization',
            python_requires='>=3.10',
            packages=find_packages(),
            include_package_data=True,
            setup_rquires=["pip>=24.2", "setuptools>=73.0.0", "numpy==1.23.0", "wheel", "cython"],
            install_requires=None,
            dependency_links=None,
            project_urls={
                'Source': 'https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-benchmark/tools',
                'Bug Reports': 'https://e2e.ti.com/support/processors-group/processors/tags/TIDL',
            },
        )
    except:
        traceback.print_exc()
        raise RuntimeError('tidl-tools-package - setup failed')
    #
    print("INFO: tidl-tools-package - setup done")


def get_arg_parser():
    tools_version_default="10.1"
    parser = argparse.ArgumentParser()
    parser.add_argument('setup_command', type=str, default=tools_version_default)
    parser.add_argument('--tools_version', type=str, default=tools_version_default)
    parser.add_argument('--tools_type', type=str, default="")
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    # setup doesn't recognize the args of this script - remove them
    keep_entries = ["develop", "install"]
    for arg in sys.argv[1:]:
        if not any([arg.startswith(r) for r in keep_entries]):
            sys.argv.remove(arg)
        #
    #

    tools_version = args.tools_version
    tools_type = args.tools_type
    tools_type = "" if tools_type.lower()=="cpu" else tools_type.lower()
    tools_type = "_gpu" if tools_type.lower()=="gpu" else tools_type.lower()

    main(tools_version, tools_type)