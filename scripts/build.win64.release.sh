#!/bin/bash
#
# scripts/build.win64.debug.sh <cmake-args...>
#
# Toolchain release build for mingw-w64 arch using MXE (https://mxe.cc/)
#
# Required environment variables:
# MXE_ROOT=<mxe-root-dir>
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: ${CMAKE_SOURCE_DIR}/_${ARCH}.build/Release]
#  INSTALL_PATH: Directory (prefix) to install to [default: ${CMAKE_SOURCE_DIR}/_${ARCH}.install]
#  NUM_PROCS: Number of processors to build with [default: attempt to find #procs]
#  OPT_DOC - enable documentation build [default: off].
#  OPT_BLAS_INT64 - enable armadillo 64-bit integer support [default: off]
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

ARCH=win64
FULL_ARCH=x86_64-w64-mingw32
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LOCAL_SCRIPTS_CONFIG_FILE=${LOCAL_SCRIPTS_CONFIG_FILE:-${SCRIPT_DIR}/../config/cmake-build-scripts.conf}
[ -f ${LOCAL_SCRIPTS_CONFIG_FILE} ] && . ${LOCAL_SCRIPTS_CONFIG_FILE}
SRC_PATH=${SCRIPT_DIR}/..
TOOLCHAIN_FILE=${SRC_PATH}/cmake/UncommonCMakeModules/Toolchains/Toolchain-MXE-${FULL_ARCH}.cmake
if [ ! -f $TOOLCHAIN_FILE ]; then
    echo "Unable to find toolchain file: '$TOOLCHAIN_FILE' for arch '$ARCH'"
    exit 1
fi

INSTALL_PATH=${INSTALL_PATH:-${SRC_PATH}/_${ARCH}.install}
BUILD_PATH=${BUILD_PATH:-${SRC_PATH}/_${ARCH}.build/Release}
OPT_DOC=${OPT_DOC:-Off}
OPT_BLAS_INT64=${OPT_BLAS_INT64:-Off}

ARGS="-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE"
ARGS="${ARGS} -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=ON"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_DOC=${OPT_DOC}"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=On"
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES=On"
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES_BUILD_TREE=On"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_WIN64}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_RELEASE}"
ARGS="${ARGS} -DOPT_BLAS_INT64=${OPT_BLAS_INT64}"

set -ex
#rm -rf $BUILD_PATH
cmake -H${SRC_PATH} -B$BUILD_PATH  -DCMAKE_BUILD_TYPE=Release ${ARGS} ${CMAKE_EXTRA_ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target all -- -j$NUM_PROCS
if [ "${OPT_DOC,,}" == "on" ] || [ $OPT_DOC -eq 1 ]; then
    VERBOSE=1 cmake --build $BUILD_PATH --target pdf -- -j$NUM_PROCS
fi
VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j$NUM_PROCS
