# PriorHessian - Main CMakeLists.txt
# A library for fast computation of log-likelihoods, and derivatives, of multivariate priors defined with Archemedian copulas.
#
# Mark J. Olah (mjo@cs.unm DOT edu)
# Copyright 2017-2019
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
# See: LICENCE file

cmake_minimum_required( VERSION 3.9 )

message(STATUS "ENV{FC}:$ENV{FC}")
message(STATUS "CMAKE_Fortran_COMPILER:${CMAKE_Fortran_COMPILER}")
project(PriorHessian VERSION 0.2 LANGUAGES CXX Fortran)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    option(BUILD_TESTING "Build testing framework" ON)
else()
    option(BUILD_TESTING "Build testing framework" OFF)
endif()
option(OPT_DOC "Build documentation" OFF)
option(OPT_INSTALL_TESTING "Install testing executables" OFF)
option(OPT_EXPORT_BUILD_TREE "Configure the package so it is usable from the build tree.  Useful for development." OFF)
option(OPT_BLAS_INT64 "Use 64-bit integers for Armadillo, BLAS, and LAPACK." OFF)

if(NOT BUILD_SHARED_LIBS AND NOT BUILD_STATIC_LIBS)
  set (BUILD_SHARED_LIBS ON)  #Must build at least one of SHARED_ and STATIC_LIBS.  Default SHARED_
endif()

message(STATUS "OPTION: BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "OPTION: BUILD_STATIC_LIBS: ${BUILD_STATIC_LIBS}")
message(STATUS "OPTION: BUILD_TESTING: ${BUILD_TESTING}")
message(STATUS "OPTION: OPT_DOC: ${OPT_DOC}")
message(STATUS "OPTION: OPT_INSTALL_TESTING: ${OPT_INSTALL_TESTING}")
message(STATUS "OPTION: OPT_EXPORT_BUILD_TREE: ${OPT_EXPORT_BUILD_TREE}")
message(STATUS "OPTION: OPT_BLAS_INT64: ${OPT_BLAS_INT64}")

#Add UcommonCmakeModules git subpreo to path.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake/UncommonCMakeModules)

### Dependencies
include(AddExternalDependency)
#BacktraceException allows for exceptions that encode a backtrace for debugging
add_external_dependency(NAME BacktraceException URL https://github.com/markjolah/BacktraceException.git VARS OPT_INSTALL_TESTING)

# Boost configure.  We require boost::math
find_package(Boost REQUIRED)

# Find Blas and Lapack using pkg-config in cross-compilation aware way.  Using local FindBLAS.cmake and FindLAPACK.cmake.
if(OPT_BLAS_INT64)
    set(BLAS_INT_COMPONENT BLAS_INT64)
else()
    set(BLAS_INT_COMPONENT BLAS_INT32)
endif()
find_package(BLAS REQUIRED COMPONENTS ${BLAS_INT_COMPONENT})
find_package(LAPACK REQUIRED COMPONENTS ${BLAS_INT_COMPONENT})
find_package(Armadillo REQUIRED COMPONENTS BLAS LAPACK CXX11 ${BLAS_INT_COMPONENT})
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS ${ARMADILLO_PRIVATE_COMPILE_DEFINITIONS})


### CFLAGS
#Configure standard CFlags and definitions for debug builds
include(ConfigureDebugBuilds)

### Source directories
add_subdirectory(src)

### PackageConfig Exports
include(ExportPackageWizzard)
set(EXPORTED_FIND_MODULES cmake/UncommonCMakeModules/FindArmadillo.cmake
                          cmake/UncommonCMakeModules/FindBLAS.cmake
                          cmake/UncommonCMakeModules/FindLAPACK.cmake
                          cmake/UncommonCMakeModules/MakePkgConfigTarget.cmake)
export_package_wizzard(PROVIDED_COMPONENTS ${BLAS_INT_COMPONENT} #Propogate BLAS integer size dependency
                       FIND_MODULES ${EXPORTED_FIND_MODULES})

### Testing
if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()

### Documentation
if(OPT_DOC)
    add_subdirectory(doc)
endif()
