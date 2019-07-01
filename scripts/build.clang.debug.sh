##!/bin/bash
# build.clang.debug.sh <cmake args ...>
#
# build with clang++ and run unit tests with CTest
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: ${CMAKE_SOURCE_DIR}/_build/ClangDebug]
#  INSTALL_PATH: Directory (prefix) to install to [default: ${CMAKE_SOURCE_DIR}/_install]
#  NUM_PROCS: Number of processors to build with [default: attempt to find #procs]
#  CLANG_PATH: Full path bin directory containing clang and clang++ executables.  If executables are not names
#              'clang' or 'clang++' then use the 'CLANG_CC" and "CLANG_CXX" flags for the full path.
#  CLANG_CXX: Full path to clang++ executable,  If set overrides CLANG_PATH for setting of CXX.
#  CLANG_CC: Full path to clang executable. If set overrides CLANG_PATH for setting of CC.
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
SRC_PATH=${SCRIPT_DIR}/..
LOCAL_SCRIPTS_CONFIG_FILE=${LOCAL_SCRIPTS_CONFIG_FILE:-${SCRIPT_DIR}/../config/cmake-build-scripts.conf}
[ -f ${LOCAL_SCRIPTS_CONFIG_FILE} ] && . ${LOCAL_SCRIPTS_CONFIG_FILE}
INSTALL_PATH=${INSTALL_PATH:-${SRC_PATH}/_install}
BUILD_PATH=${BUILD_PATH:-${SRC_PATH}/_build/ClangDebug}
CLANG_PATH=${CLANG_PATH:-$(dirname $(which clang++))}
[ ! -x "$CLANG_CC" ] && CLANG_CC=$CLANG_PATH/clang
[ ! -x "$CLANG_CXX" ] && CLANG_CXX=$CLANG_PATH/clang++
ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=ON"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DOPT_DOC=Off"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=On"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_CLANG}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_DEBUG}"

set -ex
#rm -rf $BUILD_PATH
CC=$CLANG_CC CXX=$CLANG_CXX cmake -H$SRC_PATH -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug ${ARGS} ${CMAKE_EXTRA_ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target all -- -j$NUM_PROCS
VERBOSE=1 cmake --build $BUILD_PATH/test --target test -- -j$NUM_PROCS
