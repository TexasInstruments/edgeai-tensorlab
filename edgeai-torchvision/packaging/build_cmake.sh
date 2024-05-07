#!/bin/bash
set -ex

PARALLELISM=8
if [ -n "$MAX_JOBS" ]; then
    PARALLELISM=$MAX_JOBS
fi

if [[ "$(uname)" != Darwin && "$OSTYPE" != "msys" ]]; then
    eval "$(./conda/bin/conda shell.bash hook)"
    conda activate ./env
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=conda
setup_env
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_plain_constraint

if [[ "$OSTYPE" == "msys" ]]; then
    conda install -yq conda-build cmake future
    pip install dataclasses
fi

setup_visual_studio_constraint
setup_junit_results_folder

if [[ "$(uname)" == Darwin ]]; then
  # TODO: this can be removed as soon as mkl's CMake support works with clang
  #  see https://github.com/pytorch/vision/pull/4203 for details
  MKL_CONSTRAINT='mkl==2021.2.0'
else
  MKL_CONSTRAINT=''
fi

if [[ $CONDA_BUILD_VARIANT == "cpu" ]]; then
  PYTORCH_MUTEX_CONSTRAINT='pytorch-mutex=1.0=cpu'
else
  PYTORCH_MUTEX_CONSTRAINT=''
fi

conda install -yq \pytorch=$PYTORCH_VERSION $CONDA_CUDATOOLKIT_CONSTRAINT $PYTORCH_MUTEX_CONSTRAINT $MKL_CONSTRAINT numpy -c nvidia -c "pytorch-${UPLOAD_CHANNEL}"
TORCH_PATH=$(dirname $(python -c "import torch; print(torch.__file__)"))

if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
    conda install -yq libpng jpeg
else
    yum install -y libpng-devel libjpeg-turbo-devel
fi

if [[ "$OSTYPE" == "msys" ]]; then
    source .circleci/unittest/windows/scripts/set_cuda_envs.sh
fi

mkdir cpp_build
pushd cpp_build

# Generate libtorchvision files
cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch -DWITH_CUDA=$CMAKE_USE_CUDA

# Compile and install libtorchvision
if [[ "$OSTYPE" == "msys" ]]; then
    "$script_dir/windows/internal/vc_env_helper.bat" "$script_dir/windows/internal/build_cmake.bat" $PARALLELISM
    CONDA_PATH=$(dirname $(which python))
    cp -r "C:/Program Files (x86)/torchvision/include/torchvision" $CONDA_PATH/include
else
    make -j$PARALLELISM
    make install

    if [[ "$(uname)" == Darwin ]]; then
        CONDA_PATH=$(dirname $(dirname $(which python)))
        cp -r /usr/local/include/torchvision $CONDA_PATH/include/
        export C_INCLUDE_PATH=/usr/local/include
        export CPLUS_INCLUDE_PATH=/usr/local/include
    fi
fi

popd

# Install torchvision locally
python setup.py develop

# Trace, compile and run project that uses Faster-RCNN
pushd test/tracing/frcnn
mkdir build

# Trace model
python trace_model.py
cp fasterrcnn_resnet50_fpn.pt build

cd build
cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch -DWITH_CUDA=$CMAKE_USE_CUDA
if [[ "$OSTYPE" == "msys" ]]; then
    "$script_dir/windows/internal/vc_env_helper.bat" "$script_dir/windows/internal/build_frcnn.bat" $PARALLELISM
    mv fasterrcnn_resnet50_fpn.pt Release
    cd Release
    export PATH=$(cygpath -w "C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64"):$(cygpath -w "C:/Program Files (x86)/torchvision/bin"):$(cygpath -w $TORCH_PATH)/lib:$PATH
else
    make -j$PARALLELISM
fi

# Run traced program
./test_frcnn_tracing

# Compile and run the CPP example
popd
cd examples/cpp/hello_world
mkdir build

# Trace model
python trace_model.py
cp resnet18.pt build

cd build
cmake .. -DTorch_DIR=$TORCH_PATH/share/cmake/Torch

if [[ "$OSTYPE" == "msys" ]]; then
    "$script_dir/windows/internal/vc_env_helper.bat" "$script_dir/windows/internal/build_cpp_example.bat" $PARALLELISM
    mv resnet18.pt Release
    cd Release
else
    make -j$PARALLELISM
fi

# Run CPP example
./hello-world
