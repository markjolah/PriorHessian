##!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_PATH=${SCRIPT_DIR}/..
INSTALL_PATH=${SRC_PATH}/_install
BUILD_PATH=${SRC_PATH}/_build/ClangDebug
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
COMMON_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
CLANG_VERSION=8


rm -rf $INSTALL_PATH $BUILD_PATH


set -ex
CLANG_PATH=${CLANG_PATH:-/usr/lib64/llvm/${CLANG_VERSION}/bin}
CC=$CLANG_PATH/clang CXX=$CLANG_PATH/clang++ cmake -H$SRC_PATH -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug ${COMMON_ARGS}
VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j${NUM_PROCS}
