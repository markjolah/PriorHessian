#!/bin/bash
# scripts/doc-build.sh <cmake args ...>
# Build documentation into the build tree
# Works with Travis CI.
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: ${CMAKE_SOURCE_DIR}/_build/documentation]
#  INSTALL_PATH: Directory (prefix) to install to [default: ${CMAKE_SOURCE_DIR}/_install.documentation]
#  NUM_PROCS: Number of processors to build with [default: attempt to find #procs]

#Cross-platform get number of processors
if [ -z "$NUM_PROCS" ] || [ -n "${NUM_PROCS//[0-9]}" ] || [ ! "$NUM_PROCS" -ge 1 ]; then
    case $(uname -s) in
        Linux*) NUM_PROCS=$(grep -c ^processor /proc/cpuinfo);;
        Darwin*) NUM_PROCS=$(sysctl -n hw.logicalcpu);;
        *) NUM_PROCS=1
    esac
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_PATH=${SCRIPT_DIR}/..
BUILD_PATH=${BUILD_PATH:-${SCRIPT_DIR}/../_build/documentation}
INSTALL_PATH=${INSTALL_PATH:-${SCRIPT_DIR}/../_install.documentation}
ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DOPT_DOC=On"

set -ex
#rm -rf $BUILD_PATH
cmake -H$SRC_PATH -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -Wdev $ARGS $@
VERBOSE=1 cmake --build $BUILD_PATH --target doc -- -j$NUM_PROCS
