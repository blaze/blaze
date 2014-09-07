#!/bin/bash

# script to build blaze locally and push to binstar

#buld py26 pkg
echo "Building py26 pkg"
CONDA_PY=26 conda build conda.recipe --quiet;

#buld py27 pkg
echo "Building py27 pkg"
CONDA_PY=27 conda build conda.recipe --quiet;

#buid py33 pkg
echo "Building py33 pkg"
CONDA_PY=33 conda build conda.recipe --quiet;

#buid py34 pkg
echo "Building py34 pkg"
CONDA_PY=34 conda build conda.recipe --quiet;

CONDA_ENV=`conda info --json | jsawk 'return this.root_prefix'`
PLATFORM=`conda info --json | jsawk 'return this.platform'`
BUILD_PATH=$CONDA_ENV/conda-bld/$PLATFORM

#echo build path: $BUILD_PATH

version=`git describe --tags`
u_version=`echo $version | tr "-" _`

conda convert -p all -f $BUILD_PATH/blaze*$u_version*.tar.bz2;



#upload conda pkgs to binstar
array=(osx-64 linux-64 win-64 linux-32 win-32)
for i in "${array[@]}"
do
    echo Uploading: $i;
	binstar upload -u blaze $i/blaze*$u_version*.tar.bz2 --force;
done

#create and upload pypi pkgs to binstar

#zip is currently not working

BLAZE_DEV_VERSION=$u_version python setup.py sdist --formats=gztar
binstar upload -u blaze dist/blaze*$u_version* --package-type pypi --force;

echo "I'm done uploading"

#clean up
for i in "${array[@]}"
do
    rm -rf $i
done

rm -rf dist/

rm __conda_version__.txt MANIFEST

#####################
#Removing on binstar#
#####################


# remove entire release
# binstar remove user/package/release
# binstar --verbose remove blaze/blaze/0.4.5.dev.20140602

# remove file
# binstar remove user[/package[/release/os/[[file]]]]
# binstar remove blaze/blaze/0.4.5.dev.20140602/linux-64/blaze-0.4.5.dev.20140602-np18py27_1.tar.bz2

# show files
# binstar show user[/package[/release/[file]]]
# binstar show blaze/blaze/0.4.5.dev.20140604
