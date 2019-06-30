#!/bin/bash
# build.sh <cmake-args...>
#
# PriorHessian default build script.
#
# Release-only build to local install prefix with build-tree export support.
# Cleans up build directories only.
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: _build/Release]
#  INSTALL_PATH: Directory (prefix) to install to [default: _install]
#  NUM_PROCS: Number of processors to build with: [default: attempt to find #procs]
#

#Cross-platform get number of processors
if [ -z "$NUM_PROCS" ] || [ -n "${NUM_PROCS//[0-9]}" ] || [ ! "$NUM_PROCS" -ge 1 ]; then
    case $(uname -s) in
        Linux*) NUM_PROCS=$(grep -c ^processor /proc/cpuinfo);;
        Darwin*) NUM_PROCS=$(sysctl -n hw.logicalcpu);;
        *) NUM_PROCS=1
    esac
fi

INSTALL_PATH=${INSTALL_PATH:-_install}
BUILD_PATH=${BUILD_PATH:-_build/Release}

ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=ON"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DOPT_DOC=Off"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=On"
ARGS="${ARGS} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=On" #Otherwise dependencies found in build directories won't be found in install tree unless LD_LIBRARY_PATH is modified
ARGS="${ARGS} -DOPT_BLAS_INT64=ON"

set -ex
rm -rf $BUILD_PATH
cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Release ${ARGS}
cmake --build $BUILD_PATH --target install -- -j${NUM_PROCS}
