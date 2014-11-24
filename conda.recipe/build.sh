#!/bin/bash

BLD_DIR=`pwd`

# Recipe and source are stored together
SRC_DIR=$RECIPE_DIR/..
pushd $SRC_DIR

# X.X.X.dev builds
u_version=`git describe --tags | $PYTHON $SRC_DIR/conda.recipe/version.py`

echo $u_version > __conda_version__.txt

cp __conda_version__.txt $BLD_DIR

$PYTHON setup.py install
popd

