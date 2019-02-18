#!/bin/bash
# install-blas-lapack-int64.sh <INSTALL_PREFIX> <CMAKE-ARGS...>
#
# Download, configure and install armadillo.  If install prefix is omitted, defaults to root.
#
if [ -z "$1" ]; then
    INSTALL_PREFIX="/"
else
    if [ ! -d "$1" ]; then
        mkdir -p $1
    fi
    INSTALL_PREFIX=$(cd $1; pwd)
fi

if [ -z "$FC" ]; then
    FC=$(find /usr/bin -name gfortran* -print -quit)
fi

WORK_DIR=_work
PKG_NAME=LAPACK
BUILD_PATH=_build
PKG_URL="https://github.com/Reference-LAPACK/lapack-release.git"
PKG_BRANCH="lapack-3.8.0"
FFLAGS="-fdefault-integer-8 $FFLAGS" #Force 64-bit integer support
NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)
REPOS_DIR=$WORK_DIR/$PKG_NAME

CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
CMAKE_ARGS="${CMAKE_ARGS} -DBUILD_SHARED_LIBS=On -DBUILD_STATIC_LIBS=Off"
CMAKE_ARGS="${CMAKE_ARGS} -DCBLAS=OFF -DLAPACKE=OFF"
CMAKE_ARGS="${CMAKE_ARGS} ${@:2}"
if [ -d "${REPOS_DIR}" ]; then
    rm -rf $REPOS_DIR
fi
echo "using CMAKE_ARGS:${CMAKE_ARGS}"

set -ex
${FC} --version
mkdir -p $REPOS_DIR
cd $WORK_DIR
git clone $PKG_URL -b $PKG_BRANCH $PKG_NAME --depth 1
cd $PKG_NAME
if [ ! -d "$(pwd)/$BUILD_PATH" ]; then
    mkdir -p $BUILD_PATH
fi
echo "$(pwd)"
echo "BUILD_PATH:${BUILD_PATH}"
cmake . -B$BUILD_PATH -DCMAKE_Fortan_COMPILER="$FC" -DCMAKE_Fortran_FLAGS="${FFLAGS}"  ${CMAKE_ARGS}
cd $BUILD_PATH
make all -j$NUM_PROCS
sudo make install
