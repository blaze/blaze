#!/bin/bash

BLD_DIR=`pwd`

# Recipe and source are stored together
SRC_DIR=$RECIPE_DIR/..
pushd $SRC_DIR

# X.X.X.dev builds
version=`git describe --tags`
u_version=`echo $version | tr "-" _`

echo $u_version> __conda_version__.txt

cp __conda_version__.txt $BLD_DIR

$PYTHON setup.py install
popd

