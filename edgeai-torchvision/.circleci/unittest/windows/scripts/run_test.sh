#!/usr/bin/env bash

set -e

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "$this_dir/set_cuda_envs.sh"

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
pytest --cov=torchvision --junitxml=test-results/junit.xml -v --durations 20 test --ignore=test/test_datasets_download.py
