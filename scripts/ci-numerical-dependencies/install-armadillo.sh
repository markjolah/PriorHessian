#!/bin/bash
# install-armadillo.sh <INSTALL_PREFIX>
#
# Download, configure and install armadillo.  If install prefix is omitted, defaults to root.
#
if [ -z "$1" ]; then
    INSTALL_PREFIX="/usr"
    SUDO=sudo
else
    if [ ! -d "$1" ]; then
        mkdir -p $1
    fi
    INSTALL_PREFIX=$(cd $1; pwd)
    SUDO=""
fi

WORK_DIR=_work
PKG_NAME=armadillo
BUILD_PATH=_build
PKG_URL="https://gitlab.com/conradsnicta/armadillo-code.git"
PKG_BRANCH="9.300.x"
NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)
REPOS_DIR=$WORK_DIR/$PKG_NAME

CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
CMAKE_ARGS="${CMAKE_ARGS} ${@:2}"

set -ex

rm -rf $REPOS_DIR
mkdir -p $REPOS_DIR
cd $WORK_DIR
git clone $PKG_URL -b $PKG_BRANCH $PKG_NAME --depth 1
cd $PKG_NAME
mkdir -p $BUILD_PATH
cmake . -B$BUILD_PATH ${CMAKE_ARGS}
cd $BUILD_PATH
make all -j$NUM_PROCS
$SUDO make install
