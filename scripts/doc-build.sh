#!/bin/bash
# scripts/doc-build.sh <cmake args ...>
# Build documentation into the build tree
# Works with Travis CI.
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: ${CMAKE_SOURCE_DIR}/_build/documentation]
#  INSTALL_PATH: Directory (prefix) to install to [default: ${CMAKE_SOURCE_DIR}/_install.documentation]
#  NUM_PROCS: Number of processors to build with [default: attempt to find #procs]
#  CMAKE_EXTRA_ARGS: Extra CMAKE arguments passed as environment variable
#  LOCAL_SCRIPTS_CONFIG_FILE: Path to configuration file for local options for the cmake-build-scripts subrepo

#Cross-platform get number of processors
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
BUILD_PATH=${BUILD_PATH:-${SCRIPT_DIR}/../_build/documentation}
INSTALL_PATH=${INSTALL_PATH:-${SCRIPT_DIR}/../_install.documentation}
ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DOPT_DOC=On"
ARGS="${ARGS} -DBUILD_TESTING=Off"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=Off"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=Off"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_DOC}"

set -ex
#rm -rf $BUILD_PATH
cmake -H$SRC_PATH -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -Wdev ${ARGS} ${CMAKE_EXTRA_ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target doc -- -j$NUM_PROCS
