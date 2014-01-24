#!/usr/bin/env bash

# Make sure the package information is up-to-date
apt-get update

# Compilers
apt-get install -y g++-4.7
apt-get install -y gfortran-4.7
apt-get install -y clang-3.4

# Configuration
apt-get install -y cmake

# Source control
apt-get install -y git

# Anaconda Python (miniconda3) with Python dependencies
echo Downloading Miniconda3...
curl -O http://repo.continuum.io/miniconda/Miniconda3-2.2.2-Linux-x86_64.sh
su -c 'bash Miniconda3-*.sh -b' vagrant
# Add the dev channel
printf 'channels:\n - http://repo.continuum.io/pkgs/dev\n - defaults\n' > .condarc
chown vagrant .condarc
# Install dependencies
su -c '~/anaconda/bin/conda install --yes ipython llvmpy cython numba numpy scipy dynd-python llvmmath ply pycparser pyparsing pyyaml flask nose' vagrant
# Add anaconda to the PATH
printf '\nexport PATH=~/anaconda/bin:$PATH\n' >> .bashrc
chown vagrant .bashrc

# Clone and install various projects
for PROJ in datashape blz blaze
do
    git clone https://github.com/ContinuumIO/${PROJ}.git
    chown -R vagrant ${PROJ}
    pushd ${PROJ}
    su -c '~/anaconda/bin/python setup.py install' vagrant
    popd
done

git clone https://github.com/pykit/pykit.git
chown -R vagrant pykit
pushd pykit
# TEMPORARY HACK, use 0.1.0 for pykit because it has Python 3 bugs
su -c 'git checkout 0.1.0' vagrant
su -c '~/anaconda/bin/python setup.py install' vagrant
popd

