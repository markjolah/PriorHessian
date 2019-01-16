##!/bin/bash

INSTALL_PATH=_install
BUILD_PATH=_build/Debug
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
ARGS=""
ARGS="${ARGS} -DBUILD_TESTING=On"

set -ex
rm -rf $INSTALL_PATH $BUILD_PATH
cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ${ARGS}
cmake --build $BUILD_PATH/Debug --target install -- -j${NUM_PROCS}
ctest ${PWD} $BUILD_PATH/Debug -V -j${NUM_PROCS}
