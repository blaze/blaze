#!/bin/bash

BLD_DIR=`pwd`

# Recipe and source are stored together
SRC_DIR=$RECIPE_DIR/..
pushd $SRC_DIR

# YYYYMMDD.X.X.X.dev builds
arr=$(git describe --tags | tr "-" " ")
IFS=' ' read -a gitarray <<< "${arr}"
version=${gitarray[0]}
commithash=${gitarray[2]}

echo $version, $commithash

date=`date "+%Y%m%d"`

echo ${version}_${commithash} > __conda_version__.txt
cp __conda_version__.txt $BLD_DIR

$PYTHON setup.py install
popd

