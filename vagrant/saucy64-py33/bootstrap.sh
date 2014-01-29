#!/usr/bin/env bash

# Make sure the package information is up-to-date
apt-get update || exit 1

# Compilers
apt-get install -y g++-4.7 || exit 1
apt-get install -y gfortran-4.7 || exit 1
apt-get install -y clang-3.4 || exit 1

# Configuration
apt-get install -y cmake || exit 1

# Source control
apt-get install -y git || exit 1

# Anaconda Python (miniconda3) with Python dependencies
echo Downloading Miniconda3...
curl -O http://repo.continuum.io/miniconda/Miniconda3-3.0.0-Linux-x86_64.sh || exit 1
su -c 'bash Miniconda3-*.sh -b -p ~/anaconda' vagrant || exit 1
# Install dependencies
su -c '~/anaconda/bin/conda install --yes ipython llvmpy cython numba numpy scipy llvmmath ply pycparser pyparsing pyyaml flask nose pytables' vagrant || exit 1
# Add anaconda to the PATH
printf '\nexport PATH=~/anaconda/bin:$PATH\n' >> .bashrc
chown vagrant .bashrc
export PATH=~/anaconda/bin:$PATH

# Clone and install dynd-python
git clone https://github.com/ContinuumIO/dynd-python.git || exit 1
mkdir dynd-python/libraries
pushd dynd-python/libraries
git clone https://github.com/ContinuumIO/libdynd.git || exit 1
popd
mkdir dynd-python/build
chown -R vagrant dynd-python
pushd dynd-python/build
su -c 'cmake -DPYTHON_EXECUTABLE=~/anaconda/bin/python -DCYTHON_EXECUTABLE=~/anaconda/bin/cython ..' vagrant || exit 1
su -c 'make' vagrant || exit 1
make install || exit 1
ldconfig
popd

# Clone and install various projects
for PROJ in datashape blz blaze
do
    git clone https://github.com/ContinuumIO/${PROJ}.git || exit 1
    chown -R vagrant ${PROJ}
    pushd ${PROJ}
    su -c '~/anaconda/bin/python setup.py install' vagrant || exit 1
    popd
done

# Clone and install pykit
git clone https://github.com/pykit/pykit.git || exit 1
chown -R vagrant pykit
pushd pykit
su -c '~/anaconda/bin/python setup.py install' vagrant || exit 1
popd

