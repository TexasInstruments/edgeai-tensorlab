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
import shutil
import glob

models_path = './models'

# Model file extensions and related files
model_extensions = ['.onnx', '_model.onnx', '.pb', '_model.pb', '.tflite', '_model.tflite', '.caffemodel', '_model.caffemodel', '.prototxt', '.params', '_model.params', '.log', '_checkpoint.pth', '_model.pth', '.pth', '.json', '_result.log', '_readme.txt', '_license.txt']
model_extensions += [m+'.link' for m in model_extensions]

config_extensions = ['_config.yaml', '_model_config.yaml']
config_extensions += [c+'.link' for c in config_extensions]


def check_is_model_file(file_name):
    is_model_file = any(file_name.endswith(ext) for ext in model_extensions)
    return is_model_file


def get_config_path(file_path):
    """Check if a config file exists for the given model name in the same directory"""
    for model_ext in model_extensions:
        if file_path.endswith(model_ext):
            file_path = file_path[:-len(model_ext)]
            break
    for config_ext in config_extensions:
        config_path = file_path + config_ext  
        if os.path.exists(config_path):
            return config_path
    return None

def cleanup_model_files(dry_run=True):
    """
    Clean up model files that don't have corresponding config files
    
    Args:
        dry_run (bool): If True, only print what would be removed without actually removing
    """
    if not os.path.exists(models_path):
        print(f"Models path '{models_path}' does not exist!")
        return
    
    print(f"Scanning for model files in: {models_path}")
    
    files_to_remove = []
    files_to_keep = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(models_path):
        for file in files:
            filepath = os.path.join(root, file)
            
            # Check if this is a model file (candidate for removal)
            is_model_file = check_is_model_file(file)
            
            if is_model_file:
                # Check if corresponding config file exists in the same directory
                config_path = get_config_path(filepath)
                if config_path:
                    files_to_keep.append(filepath)
                    print(f"KEEP: {filepath} (has config: {config_path})")
                else:
                    files_to_remove.append(filepath)
                    print(f"REMOVE: {filepath} (no config found)")
            else:
                # Not a model file, so keep it
                files_to_keep.append(filepath)
    
    # Report findings
    print(f"\nSummary:")
    print(f"  Total files scanned: {len(files_to_keep) + len(files_to_remove)}")
    print(f"  Model files to keep: {len([f for f in files_to_keep if any(f.endswith(ext) for ext in model_extensions)])}")
    print(f"  Model files to remove: {len(files_to_remove)}")
    print(f"  Other files (kept): {len([f for f in files_to_keep if not any(f.endswith(ext) for ext in model_extensions)])}")
    
    if files_to_remove:
        print(f"\nModel files that would be removed:")
        for file in sorted(files_to_remove):
            print(f"  - {file}")
    else:
        print(f"\nNo model files found without config files!")
    
    # Actually remove files if not in dry_run mode
    if not dry_run and files_to_remove:
        print(f"\nRemoving {len(files_to_remove)} files...")
        for file in files_to_remove:
            try:
                os.remove(file)
                print(f"  Removed: {file}")
            except Exception as e:
                print(f"  Error removing {file}: {e}")
    elif dry_run and files_to_remove:
        print(f"\nDRY RUN: Would remove {len(files_to_remove)} files")
    
    return files_to_remove, files_to_keep

if __name__ == "__main__":
    # Run in dry-run mode by default for safety
    cleanup_model_files(dry_run=False)

