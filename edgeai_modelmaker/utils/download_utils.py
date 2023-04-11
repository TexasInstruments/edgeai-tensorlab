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
import shutil
import urllib
import gzip
import tarfile
import zipfile
import requests
import copy
import warnings

from . import misc_utils


def copy_file(file_path, file_path_local):
    if os.path.realpath(file_path) != os.path.realpath(file_path_local):
        os.makedirs(os.path.dirname(file_path_local), exist_ok=True)
        shutil.copy2(file_path, file_path_local)
    #
    return file_path_local


def extract_files(download_file, extract_root):
    extract_success = False
    if extract_root is None:
        extract_root = os.path.dirname(download_file)
    #
    if download_file.endswith('.tar'):
        with tarfile.open(download_file, 'r') as tar:
            tar.extractall(path=extract_root)
        #
        extract_success = True
    elif download_file.endswith('.tar.gz') or download_file.endswith('.tgz'):
        with tarfile.open(download_file, 'r:gz') as tar:
            tar.extractall(path=extract_root)
        #
        extract_success = True
    elif download_file.endswith('.gz'):
        to_path = os.path.join(extract_root, os.path.splitext(os.path.basename(download_file))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(download_file) as zip_f:
            out_f.write(zip_f.read())
        #
        extract_success = True
    elif download_file.endswith('.zip'):
        with zipfile.ZipFile(download_file, 'r') as z:
            z.extractall(extract_root)
        #
        extract_success = True
    #
    return extract_success


def download_url(dataset_url, download_root, save_filename=None, progressbar_creator=None):
    if not isinstance(dataset_url, str) or not (dataset_url.startswith('http://') or dataset_url.startswith('https://')):
        return True, '', dataset_url
    #

    download_success = False
    exception_message = ''
    download_path = None

    try:
        save_filename = save_filename if save_filename else os.path.basename(dataset_url)
        download_file = os.path.join(download_root, save_filename)
        if not os.path.exists(download_file):
            print(f'downloading from {dataset_url} to {download_file}')
            progressbar_creator = progressbar_creator or misc_utils.ProgressBar
            resp = requests.get(dataset_url, stream=True, allow_redirects=True)
            total_size = int(resp.headers.get('content-length'))
            progressbar_obj = progressbar_creator(total_size, unit='B')
            os.makedirs(download_root, exist_ok=True)
            with open(download_file, 'wb') as fp:
                for content in resp.iter_content(chunk_size=1024):
                    fp.write(content)
                    progressbar_obj.update(len(content))
                #
            #
        #
        download_path = download_file
        download_success = True
    except urllib.error.URLError as message:
        download_success = False
        exception_message = str(message)
        print(exception_message)
    except urllib.error.HTTPError as message:
        download_success = False
        exception_message = str(message)
        print(exception_message)
    except NameError as message:
        download_success = False
        exception_message = str(message)
        print(exception_message)
    # except Exception as message:
    #     # sometimes getting exception even though download succeeded.
    #     download_path = download_file
    #     download_success = True
    #     exception_message = str(message)
    #
    return download_success, exception_message, download_path


def download_and_extract(dataset_url, download_root, extract_root=None, save_filename=None, progressbar_creator=None, extract=True):
    download_success, exception_message, dataset_url = \
        download_url(dataset_url, download_root, save_filename, progressbar_creator)

    extract_root = extract_root or os.path.dirname(dataset_url)
    if extract and extract_files(dataset_url, extract_root):
        return True, '', extract_root
    else:
        return download_success, exception_message, dataset_url
    #


def download_file(dataset_url, download_root, extract_root=None, save_filename=None, progressbar_creator=None,
                  force_linkfile=True, make_symlink=True, extract=True):
    if not isinstance(dataset_url, str):
        return False, '', ''

    is_linkfile = dataset_url.endswith('.link')
    is_url = dataset_url.startswith('http://') or dataset_url.startswith('https://')
    download_root = os.path.abspath('./') if download_root is None else download_root

    if not is_url:
        if os.path.exists(dataset_url):
            save_filename = save_filename or os.path.basename(dataset_url)
            local_file = os.path.join(download_root, save_filename)
            if os.path.isfile(dataset_url):
                copy_file(dataset_url, local_file)
                dataset_url = local_file
            elif make_symlink and os.path.isdir(dataset_url):
                if dataset_url != extract_root:
                    if os.path.islink(extract_root):
                        os.unlink(extract_root)
                    #
                    misc_utils.make_symlink(dataset_url, extract_root)
                #
                dataset_url = extract_root
            #
        elif force_linkfile and not os.path.exists(dataset_url):
            url_link = dataset_url if is_linkfile else dataset_url + '.link'
            if os.path.exists(url_link):
                with open(url_link) as fp:
                    dataset_url = fp.readline().rstrip()
                #
            #
        #
    #
    return download_and_extract(dataset_url, download_root, extract_root=extract_root, save_filename=save_filename,
                                progressbar_creator=progressbar_creator, extract=extract)


def download_files(dataset_urls, download_root, extract_root=None, save_filenames=None, log_writer=None,
                   progressbar_creator=None, extract=True):
    if log_writer is not None:
        success_writer, warning_writer = log_writer[:2]
    else:
        success_writer, warning_writer = print, print
    #
    dataset_urls = dataset_urls if isinstance(dataset_urls, (list,tuple)) else [dataset_urls]
    save_filenames = save_filenames if isinstance(save_filenames, (list,tuple)) else \
        ([None]*len(dataset_urls) if save_filenames is None else [save_filenames])

    download_paths = []
    for dataset_url_id, (dataset_url, save_filename) in enumerate(zip(dataset_urls, save_filenames)):
        success_writer(f'Downloading {dataset_url_id+1}/{len(dataset_urls)}: {dataset_url}')
        download_success, message, download_path = download_file(
            dataset_url, download_root=download_root, extract_root=extract_root,
            save_filename=save_filename, progressbar_creator=progressbar_creator, extract=extract)
        if download_success:
            success_writer(f'Download done for {dataset_url}')
        else:
            warning_writer(f'Download failed for {dataset_url} {str(message)}')
        #
        download_paths.append(download_path)
    #
    return download_success, message, download_paths


def download_url_entry(download_entry, download_path=None, download_root=None):
    # fetch the pretrained checkpoint if it is a url.
    # if it is a url and the downloaded copy is present, it will be reused.
    if isinstance(download_entry, list) and bool(download_entry):
        pretrained_path_in = copy.deepcopy(download_entry)
        download_entry = download_url_entry(pretrained_path_in[0], download_path, download_root)
        for d_entry in pretrained_path_in[1:]:
            download_url_entry(d_entry, download_path, download_root)
        #
    elif isinstance(download_entry, dict) and bool(download_entry):
        download_root = download_entry['download_path'] if 'download_path' in download_entry else download_root
        download_entry = download_url_entry(download_entry['download_url'], download_path, download_root)
    elif misc_utils.is_url(download_entry):
        download_root = download_path if download_root is None else download_root
        # {download_path} is a special keyword that will be replaced with the current download path
        download_root = download_root.replace('{download_path}', download_path)
        download_success, exception_message, download_entry = download_file(
            download_entry, download_root, extract=False)
        if not download_success:
            warnings.warn(f'url could not be downloaded: {download_entry}')
            return None
        #
    elif isinstance(download_entry, str):
        print(f'assuming the given download_url is a valid path: {download_entry}')
    else:
        warnings.warn(f'unrecognized download_url: {download_entry}')
    #
    return download_entry


def _download_entry(download_entry, download_path, download_root=None):
    if download_entry:
        download_entry = download_url_entry(download_entry, download_path, download_root)
    #
    return download_entry


def _download_dataset(input_data_path, download_path):
    download_root = os.path.join(download_path, 'datasets')
    return _download_entry(input_data_path, download_path, download_root)


def _download_pretrained_checkpoint(pretrained_path, download_path, model_name):
    download_root = os.path.join(download_path, 'pretrained', model_name)
    return _download_entry(pretrained_path, download_path, download_root)


def download_all(params):
    # if the model_config has a download section, download it
    if hasattr(params, 'download'):
        _download_entry(params.download, params.common.download_path)
    #
    # download dataset
    if hasattr(params, 'dataset'):
        params.dataset.input_data_path = _download_dataset(
            params.dataset.input_data_path, params.common.download_path)
    #
    # fetch the pretrained checkpoint if it is a url.
    # if it is a url and the downloaded copy is present, it will be reused.
    if hasattr(params, 'training'):
        params.training.pretrained_checkpoint_path = _download_pretrained_checkpoint(
            params.training.pretrained_checkpoint_path,
            params.common.download_path, params.training.model_name)
    #

