#!/bin/bash
# scripts/dist-build.sh <INSTALL_DIR> <cmake-args...>
#
# Builds a re-distributable release-only build for C++ libraries.
# Testing and documentation are enabled.
# Creates a .zip and .tar.gz archives.
#
# Args:
#  <INSTALL_DIR> - path to distribution install directory [Default: ${CMAKE_SOURCE_DIR}/_dist].
#                  The distribution files will be created under this directory with names based on
#                  package and versions.
#  <cmake_args...> - additional cmake arguments.
#
# Controlling Environment Variables:
#  BUILD_PATH: Directory to build under (if existing, it will be deleted) [default: ${CMAKE_SOURCE_DIR}/_build/dist]
#  NUM_PROCS: Number of processors to build with [default: attempt to find #procs]
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_PATH=${SCRIPT_DIR}/..
LOCAL_SCRIPTS_CONFIG_FILE=${LOCAL_SCRIPTS_CONFIG_FILE:-${SCRIPT_DIR}/local-config/cmake-build-scripts.conf}
[ -f ${LOCAL_SCRIPTS_CONFIG_FILE} ] && . ${LOCAL_SCRIPTS_CONFIG_FILE}
NAME=$(grep -Po "project\(\K([A-Za-z]+)" ${SRC_PATH}/CMakeLists.txt)
VERSION=$(grep -Po "project\([A-Za-z]+ VERSION \K([0-9.]+)" ${SRC_PATH}/CMakeLists.txt)
if [ -z $NAME ] || [ -z $VERSION ]; then
    echo "Unable to find package name and version from: ${SRC_PATH}/CMakeLists.txt"
    exit 1
fi

DIST_DIR_NAME=${NAME}-${VERSION}
if [ -z $1 ]; then
    INSTALL_PATH=${SRC_PATH}/_dist/$DIST_DIR_NAME
else
    INSTALL_PATH=${INSTALL_PATH:-$1/$DIST_DIR_NAME}
fi

OPT_BLAS_INT64=${OPT_BLAS_INT64:-Off}

ZIP_FILE=${NAME}-${VERSION}.zip
TAR_FILE=${NAME}-${VERSION}.tbz2

BUILD_PATH=${BUILD_PATH:-${SRC_PATH}/_build/dist}
NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)

ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=ON"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_DOC=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=Off"
ARGS="${ARGS} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=On"  # Disable finding packages in the build-tree
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_RELEASE}"
ARGS="${ARGS} ${LOCAL_CMAKE_ARGS_DIST}"
ARGS="${ARGS} -DOPT_BLAS_INT64=$OPT_BLAS_INT64"

set -ex
#rm -rf $BUILD_PATH
cmake -H$SRC_PATH -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Release ${ARGS} ${CMAKE_EXTRA_ARGS} ${@:2}
cmake --build $BUILD_PATH --target doc -- -j$NUM_PROCS
cmake --build $BUILD_PATH --target pdf -- -j$NUM_PROCS
cmake --build $BUILD_PATH --target install -- -j$NUM_PROCS

cd $INSTALL_PATH/..
zip -rq $ZIP_FILE $DIST_DIR_NAME
tar cjf $TAR_FILE $DIST_DIR_NAME
