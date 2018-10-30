##!/bin/bash

INSTALL_PATH=_install
BUILD_PATH=_build
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
COMMON_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
rm -rf $INSTALL_PATH $BUILD_PATH/ClangDebug

set -e
CC=/usr/lib64/llvm/7/bin/clang
CXX=/usr/lib64/llvm/7/bin/clang++
cmake -H. -B$BUILD_PATH/ClangDebug -DCMAKE_BUILD_TYPE=Debug ${COMMON_ARGS}
VERBOSE=1 cmake --build $BUILD_PATH/ClangDebug --target install -- -j${NUM_PROCS}
