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

#################################################################################

# Modified from: https://github.com/pytorch/vision
# License: License: https://github.com/pytorch/vision/blob/master/LICENSE

# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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
import os.path
import sys
import shutil
import hashlib
import gzip
import re
import tarfile
from typing import Any, Callable, List, Iterable, Optional, TypeVar
from urllib.parse import urlparse
import zipfile
from tqdm import tqdm

from . import model_utils


def copy_file(file_path, file_path_local):
    if file_path != file_path_local:
        os.makedirs(os.path.dirname(file_path_local), exist_ok=True)
        shutil.copy2(file_path, file_path_local)
    #
    return file_path_local


def copy_files(file_path, file_path_local):
    file_paths = misc_utils.as_list(file_path)
    file_paths_local = misc_utils.as_list(file_path_local)
    output_files = []
    for m, lm in zip(file_paths, file_paths_local):
        output = copy_file(m, lm)
        output_files.append(output)
    #
    return output_files


def is_url(path):
    return path.startswith('http://') or path.startswith('https://')


def download_file(url, root=None, extract_root=None, filename=None, md5=None, mode=None, force_download=False, force_linkfile=True):
    if not isinstance(url, str):
        print(f"invalid file or url: {url}")
        return None

    root = os.path.abspath('./') if root is None else root

    is_linkfile = url.endswith('.link')
    path_is_url = is_url(url)
    if (not path_is_url):
        if os.path.exists(url):
            filename = filename or os.path.basename(url)
            local_file = os.path.join(root, filename)
            copy_file(url, local_file)
            url = local_file
        elif is_linkfile or force_linkfile:
            url_link = url if is_linkfile else url + '.link'
            if os.path.exists(url_link):
                with open(url_link) as fp:
                    url = fp.read().rstrip()
                #
            #
        #
    #

    fpath = download_and_extract_archive(url, root, extract_root=extract_root, filename=filename,
                                         md5=md5, mode=mode,force_download=force_download)
    return fpath


def download_files(dataset_urls, root, extract_root=None, save_filenames=None, message=''):
    dataset_urls = dataset_urls if isinstance(dataset_urls, (list,tuple)) else [dataset_urls]
    save_filenames = save_filenames if isinstance(save_filenames, (list,tuple)) else \
        ([None]*len(dataset_urls) if save_filenames is None else [save_filenames])

    download_paths = []
    for dataset_url_id, (dataset_url, save_filename) in enumerate(zip(dataset_urls, save_filenames)):
        print(f'Downloading {dataset_url_id+1}/{len(dataset_urls)}: {dataset_url}')
        download_path = download_file(dataset_url, root=root, extract_root=extract_root,
                                      filename=save_filename)
        if download_path is not None:
            print(f'Download done for {dataset_url}')
        else:
            print(f'Download failed for {dataset_url} {str(message)}')
        #
        download_paths.append(download_path)
    #
    return download_paths


def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

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
        raise RecursionError(f"Too many redirects: {max_hops + 1})")


def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


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
        print('Using downloaded and verified file: ' + fpath)
        sys.stdout.flush()
        return fpath
    #

    print('Downloading ' + url + ' to ' + fpath)
    sys.stdout.flush()

    os.makedirs(root, exist_ok=True)

    # expand redirect chain if needed
    if max_redirect_hops > 0:
        url = _get_redirect_url(url, max_hops=max_redirect_hops)
    #

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        fpath = download_file_from_google_drive(file_id, root, filename, md5)
        #print('done.')
        return fpath
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
            print('Failed download. Trying https -> http instead.'
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


def list_dir(root: str, prefix: bool = False) -> List[str]:
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root: str, suffix: str, prefix: bool = False) -> List[str]:
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def _quota_exceeded(response: "requests.models.Response") -> bool:  # type: ignore[name-defined]
    return False
    # See https://github.com/pytorch/vision/issues/2992 for details
    # return "Google Drive - Quota exceeded" in response.text


def download_file_from_google_drive(file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
        sys.stdout.flush()
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        if _quota_exceeded(response):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(response, fpath)

    return fpath


def _get_confirm_token(response: "requests.models.Response") -> Optional[str]:  # type: ignore[name-defined]
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(
    response: "requests.models.Response", destination: str, chunk_size: int = 32768,  # type: ignore[name-defined]
) -> None:
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


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


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False,
                    verbose: bool = True, mode: Optional[str] = None):
    if verbose:
        print(f'Extracting {from_path} to {to_path}')
        sys.stdout.flush()
    #
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        mode = 'r' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='data')
    elif _is_targz(from_path) or _is_tgz(from_path):
        mode = 'r:gz' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='data')
    elif _is_tarxz(from_path):
        mode = 'r:xz' if mode is None else mode
        with tarfile.open(from_path, mode) as tar:
            tar.extractall(path=to_path, filter='data')
    elif _is_gzip(from_path):
        mode = 'r' if mode is None else mode
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        mode = 'r' if mode is None else mode
        with zipfile.ZipFile(from_path, mode) as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

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


def _is_archive(from_path):
    return _is_tar(from_path) or _is_targz(from_path) or \
           _is_gzip(from_path) or _is_zip(from_path) or _is_tgz(from_path)


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


T = TypeVar("T", str, bytes)


def verify_str_arg(
    value: T, arg: Optional[str] = None, valid_values: Iterable[T] = None, custom_msg: Optional[str] = None,
) -> T:
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ("Unknown value '{value}' for argument {arg}. "
                   "Valid values are {{{valid_values}}}.")
            msg = msg.format(value=value, arg=arg,
                             valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value
