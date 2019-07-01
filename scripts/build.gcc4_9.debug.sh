#!/bin/bash
#
# scripts/build.gcc4_9.debug.sh <cmake-args...>
#
# Toolchain build for gcc-4.9 systems (i.e., Matlab R2016b-R2017b)
#
# Required environment variables:
# X86_64_GCC4_9_LINUX_GNU_ROOT=<gcc4_9_root>
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: ${CMAKE_SOURCE_DIR}/_${ARCH}.build/Debug]
#  INSTALL_PATH: Directory (prefix) to install to [default: ${CMAKE_SOURCE_DIR}/_${ARCH}.install]
#  NUM_PROCS: Number of processors to build with [default: attempt to find #procs]
#  OPT_DOC - enable documentation build [default: off].
#  OPT_BLAS_INT64 - enable armadillo 64-bit integer support [default: off]

#Cross-platform get number of processors
if [ -z "$NUM_PROCS" ] || [ -n "${NUM_PROCS//[0-9]}" ] || [ ! "$NUM_PROCS" -ge 1 ]; then
    case $(uname -s) in
        Linux*) NUM_PROCS=$(grep -c ^processor /proc/cpuinfo);;
        Darwin*) NUM_PROCS=$(sysctl -n hw.logicalcpu);;
        *) NUM_PROCS=1
    esac
fi

ARCH=gcc4_9
FULL_ARCH=x86_64-${ARCH}-linux-gnu
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LOCAL_SCRIPTS_CONFIG_FILE=${LOCAL_SCRIPTS_CONFIG_FILE:-${SCRIPT_DIR}/../config/cmake-build-scripts.conf}
[ -f ${LOCAL_SCRIPTS_CONFIG_FILE} ] && . ${LOCAL_SCRIPTS_CONFIG_FILE}
SRC_PATH=${SCRIPT_DIR}/..
TOOLCHAIN_FILE=${SRC_PATH}/cmake/UncommonCMakeModules/Toolchains/Toolchain-${FULL_ARCH}.cmake
if [ ! -f $TOOLCHAIN_FILE ]; then
    echo "Unable to find toolchain file: '$TOOLCHAIN_FILE' for arch '$ARCH'"
    exit 1
fi

INSTALL_PATH=${INSTALL_PATH:-${SRC_PATH}/_${ARCH}.install}
BUILD_PATH=${BUILD_PATH:-${SRC_PATH}/_${ARCH}.build/Debug}
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
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES_COPY_GCC_LIBS=On"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_GCC4_9}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_DEBUG}"
ARGS="${ARGS} -DOPT_BLAS_INT64=${OPT_BLAS_INT64}"

set -ex
#rm -rf $BUILD_PATH
cmake -H$SRC_PATH -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug ${ARGS} ${CMAKE_EXTRA_ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target all -- -j$NUM_PROCS
if [ "${OPT_DOC,,}" == "on" ] || [ $OPT_DOC -eq 1 ]; then
    VERBOSE=1 cmake --build $BUILD_PATH --target pdf -- -j$NUM_PROCS
fi
VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j$NUM_PROCS
