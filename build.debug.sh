#!/bin/bash
# build.debug.sh <cmake args ...>
#
# PriorHessian example debug build script.
#
# Debug-only build to local install prefix with build-tree export support;
# Cleans up build directories only.
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: _build/Release]
#  INSTALL_PATH: Directory (prefix) to install to [default: _install]
#  NUM_PROCS: Number of processors to build with [default: attempt to find #procs]
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
BUILD_PATH=${BUILD_PATH:-_build/Debug}

ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=OFF"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DOPT_DOC=Off"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=On"
ARGS="${ARGS} -DOPT_BLAS_INT64=On"

set -ex
rm -rf $BUILD_PATH
cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -Wdev ${ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target all -- -j${NUM_PROCS}
VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j${NUM_PROCS}
