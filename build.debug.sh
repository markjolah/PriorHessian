##!/bin/bash

INSTALL_PATH=_install
BUILD_PATH=_build
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
ARGS=""
ARGS="${ARGS} -DBUILD_TESTING=On"

set -ex
rm -rf $INSTALL_PATH $BUILD_PATH
cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ${ARGS}
VERBOSE=1 cmake --build $BUILD_PATH/Debug --target install -- -j${NUM_PROCS}
