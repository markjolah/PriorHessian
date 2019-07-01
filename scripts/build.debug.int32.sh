#!/bin/bash
# scripts/build.debug.int32.sh <cmake args ...
#
# Debug-only build to local install prefix with build-tree export and OPT_BLAS_INT64=Off
#
# Cleans up build directories only.
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: ${CMAKE_SOURCE_DIR}/_build/Int32.Debug]
#  INSTALL_PATH: Directory (prefix) to install to [default: ${CMAKE_SOURCE_DIR}/_install]
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LOCAL_SCRIPTS_CONFIG_FILE=${LOCAL_SCRIPTS_CONFIG_FILE:-${SCRIPT_DIR}/../config/cmake-build-scripts.conf}
[ -f ${LOCAL_SCRIPTS_CONFIG_FILE} ] && . ${LOCAL_SCRIPTS_CONFIG_FILE}
SRC_PATH=${SCRIPT_DIR}/..
INSTALL_PATH=${INSTALL_PATH:-${SRC_PATH}/_install}
BUILD_PATH=${BUILD_PATH:-${SRC_PATH}/_build/Int32.Debug}

ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=ON"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=Off"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=Off"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_DEBUG}"
ARGS="${ARGS} -DOPT_BLAS_INT64=Off" #Force 32-bit integer LAPACK support

set -ex
#rm -rf $BUILD_PATH
cmake -H$SRC_PATH -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -Wdev ${ARGS} ${CMAKE_EXTRA_ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target all -- -j$NUM_PROCS
cmake --build $BUILD_PATH/test --target test -- -j$NUM_PROCS
