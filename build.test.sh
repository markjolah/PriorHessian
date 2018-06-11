##!/bin/bash

INSTALL_PATH=_install
BUILD_PATH=_build/Debug
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
COMMON_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
# rm -rf $INSTALL_PATH $BUILD_PATH

set -e
if [ -ne $BUILD_PATH ]; then
    cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug ${COMMON_ARGS}
fi
cmake --build $BUILD_PATH/Debug --target install -- -j${NUM_PROCS}
ctest ${PWD} $BUILD_PATH/Debug -V -j${NUM_PROCS}
