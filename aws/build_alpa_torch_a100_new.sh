#!/usr/bin/env bash

set -eu

CONDA_ENV=alpa
export HOME_DIR=$HOME/nfs/alpa-torch-software
mkdir -p ${HOME_DIR}
cd ${HOME_DIR}

export CUDA_VER_SHORT=114
export CUDA_VER=11.4
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:$PATH  # to find nvcc

export NUMPY_VER=1.22.4
export NCCL_HOME=$HOME/.cupy/cuda_lib/11.4/nccl/2.11.4
export AWS_OFI_PLUGIN=/usr/local/cuda/efa

# I don't like LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=${CUDA_HOME}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/extras/CUPTI/lib64
# export CFLAGS=-I${CUDA_HOME}/include  # This helps CuPy find the right NCCL version
# export LDFLAGS=-L${CUDA_HOME}/lib  # This helps CuPy find the right NCCL version

# Reset the system cuda home
if [[ -z `readlink -f /usr/local/cuda | grep cuda-11.4` ]]; then
    sudo rm -f /usr/local/cuda
    sudo ln -sf /usr/local/cuda-11.4 /usr/local/cuda
fi


# Remove hide redundant cuda libraries and only retain the default one
cd /etc/ld.so.conf.d
for i in `find /etc/ld.so.conf.d/ -name "cuda-*.conf"`; do echo $i; sudo mv $i $i.bak; done
echo '/usr/local/cuda/targets/x86_64-linux/lib' | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig

# Remove the aws-ofi-nccl plugin (1.3.0).
# This version will cause performance issues when used with cupy and AWS Libfabric RDMA.
# Use 1.1.15 instead. 
sudo rm -rf "${AWS_OFI_PLUGIN}"

# Remove the bundled NCCL (2.12.12) from Deep Learning AMI.
# This version will cause performance issues when used with cupy and AWS Libfabric RDMA.
# Use 2.11.4 instead.
cat << EOF > /tmp/nccl_installation.txt
/usr/local/cuda/lib/libnccl.so.2.12.12
/usr/local/cuda/lib/libnccl.so
/usr/local/cuda/lib/pkgconfig/nccl.pc
/usr/local/cuda/lib/libnccl.so.2
/usr/local/cuda/lib/libnccl_static.a
/usr/local/cuda/targets/x86_64-linux/include/nccl.h
/usr/local/cuda/targets/x86_64-linux/include/nccl_net.h
EOF

for i in `cat /tmp/nccl_installation.txt`; do sudo rm -fv $i; done

# Install conda if necessary
cd $HOME
if [[ -z `command -v conda` ]]; then
    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh 
    bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh
    ~/miniconda3/bin/conda config --set auto_activate_base false
    curl -L https://cjr.host/download/config/conda.zshrc >> ~/.zshrc.local
fi

# Make conda happy
eval "$(conda shell.bash hook)"

# Create alpa environment if necessary
if [[ -z `conda env list | grep ${CONDA_ENV}` ]]; then
    conda create -y -n ${CONDA_ENV} python=3.8
    conda activate ${CONDA_ENV}

    pip3 install cmake tqdm numpy==${NUMPY_VER} scipy numba pybind11 pulp ray tensorstore flax==0.4.1 jax==0.3.5
    pip3 install cupy-cuda${CUDA_VER_SHORT} # find cuda from the system under /usr/local/cuda

    # WARNING: nccl version should match ${CUDA_HOME}/include/nccl.h
    # conda install -y -c conda-forge nccl=${NCCL_VER} fastrlock cudatoolkit=11.4

    python3 -m cupyx.tools.install_library --library nccl --cuda ${CUDA_VER}
    # the nccl will be installed to $HOME/.cupy/cuda_lib/11.4/nccl/2.11.4/

    conda deactivate
fi

# activate the virtual environment
conda activate ${CONDA_ENV}

# Check whether NCCL is install.
# If prints some instruction, follow it.
python3 -c "from cupy.cuda import nccl; print(nccl.get_version())"
# we are expect to see 21104 here

# After we have nccl, install the aws-ofi-nccl plugin
cd $HOME/nfs
[[ -d "$HOME/nfs/aws-ofi-nccl" ]] || git clone https://github.com/aws/aws-ofi-nccl.git -b v1.1.5-aws
cd aws-ofi-nccl
./autogen.sh
sudo mkdir -p "${AWS_OFI_PLUGIN}"
./configure --prefix="${AWS_OFI_PLUGIN}" --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa --with-nccl="${NCCL_HOME}" \
        --with-cuda=/usr/local/cuda
make -j 60
sudo make install


# Install the ILP solver
sudo apt install coinor-cbc -y

# Install nightly version of torch and torchdistx
pip3 uninstall -y torch torchdistx
pip3 install torchdistx --pre --extra-index-url https://download.pytorch.org/whl/nightly

# Build functorch from source
pip3 uninstall -y functorch
# rm -rf ${HOME_DIR}/functorch
cd ${HOME_DIR}
[[ -d "${HOME_DIR}/functorch" ]] || git clone https://github.com/pytorch/functorch
cd ${HOME_DIR}/functorch
python3 setup.py install

# Use the correct version of numpy
pip3 install numpy==${NUMPY_VER}



# Download source code when necessary
cd ${HOME_DIR}
[[ -d "${HOME_DIR}/alpa" ]] || git clone --recursive https://github.com/alpa-projects/alpa.git


# Setup the bazel cache directory to NFS if it's not (this could be dangerous)
[[ -d $HOME/nfs/cache/bazel ]] || mkdir -p $HOME/nfs/cache/bazel
[[ "x`readlink -f $HOME/.cache/bazel`" == x"/nfs/$USER/cache/bazel" ]] || ln -sf /nfs/$USER/cache/bazel $HOME/.cache/bazel

# Build and install jaxlib
# mv ~/.cache/bazel ~/.cache/bazel_tmp

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
ray start --head

cd ${HOME_DIR}/alpa
python3 tests/test_install.py

ray stop