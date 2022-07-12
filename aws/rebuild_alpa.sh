#!/usr/bin/env bash

set -eu

export HOME_DIR=$HOME/nfs/alpa-torch-software
export CUDA_VER=11.4
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:$PATH
export NUMPY_VER=1.22.4

# Actrivate conda environment
eval "$(conda shell.bash hook)"
conda deactivate
conda activate alpa

# See tensorflow/third_party/gpus/cuda_configure.bzl
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=${CUDA_VER}
export TF_CUDNN_VERSION=8

pip3 uninstall -y jaxlib
cd ${HOME_DIR}/alpa/build_jaxlib
python3 build/build.py --enable_cuda --dev_install --tf_path=$(pwd)/../third_party/tensorflow-alpa --cudnn_path="${CUDA_HOME}" --cuda_version=${CUDA_VER}
cd dist
pip3 install -e .

# Install Alpa
cd ${HOME_DIR}/alpa
pip3 install -e .


# Build XLA pipeline marker custom call
cd ${HOME_DIR}/alpa/alpa/pipeline_parallel/xla_custom_call_marker
git clean -xdf
bash build.sh

# use the correct version of numpy
pip3 install numpy==${NUMPY_VER} --upgrade


# Test install
# ray start --head
# 
# cd ${HOME_DIR}/alpa
# python3 tests/test_install.py
# 
# ray stop