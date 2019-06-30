##!/bin/bash
# build.test.sh <cmake args ...>
#
# build with CMake and run unit tests with CTest
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: ${CMAKE_SOURCE_DIR}/_build/Debug]
#  INSTALL_PATH: Directory (prefix) to install to [default: ${CMAKE_SOURCE_DIR}/_install]
#  NUM_PROCS: Number of processors to build with [default: attempt to find #procs]
#  CMAKE_EXTRA_ARGS: Extra CMAKE arguments passed as environment variable
#  LOCAL_SCRIPTS_CONFIG_FILE: Path to configuration file for local options for the cmake-build-scripts subrepo

if [ -z "$NUM_PROCS" ] || [ -n "${NUM_PROCS//[0-9]}" ] || [ ! "$NUM_PROCS" -ge 1 ]; then
    case $(uname -s) in
        Linux*) NUM_PROCS=$(grep -c ^processor /proc/cpuinfo);;
        Darwin*) NUM_PROCS=$(sysctl -n hw.logicalcpu);;
        *) NUM_PROCS=1
    esac
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LOCAL_SCRIPTS_CONFIG_FILE=${LOCAL_SCRIPTS_CONFIG_FILE:-${SCRIPT_DIR}/local-config/cmake-build-scripts.conf}
[ -f ${LOCAL_SCRIPTS_CONFIG_FILE} ] && . ${LOCAL_SCRIPTS_CONFIG_FILE}
SRC_PATH=${SCRIPT_DIR}/..
BUILD_PATH=${BUILD_PATH:-${SRC_PATH}/_build/Debug}
INSTALL_PATH=${INSTALL_PATH:-${BUILD_PATH}/_install} #If we need to install some dependencies, do it internally.  We are just testing.
ARGS="-DCMAKE_BUILD_TYPE=Debug"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=Off"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=Off"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_DEBUG}"

set -ex
#rm -rf $BUILD_PATH
cmake -H${SRC_PATH} -B$BUILD_PATH  -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ${ARGS} ${CMAKE_EXTRA_ARGS} $@
cmake --build $BUILD_PATH/test --target all -- -j${NUM_PROCS}
cmake --build $BUILD_PATH/test --target test -- -j${NUM_PROCS}
