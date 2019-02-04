# PriorHessian - Main CMakeLists.txt
# A library for fast computation of log-likelihoods, and derivatives, of multivariate priors defined with Archemedian copulas.
#
# Mark J. Olah (mjo@cs.unm DOT edu)
# Copyright 2017-2019
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
# See: LICENCE file

cmake_minimum_required( VERSION 3.9 )
project(PriorHessian VERSION 0.2 LANGUAGES CXX Fortran)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    option(BUILD_TESTING "Build testing framework" ON)
else()
    option(BUILD_TESTING "Build testing framework" OFF)
endif()
option(OPT_INSTALL_TESTING "Install testing executables" OFF)
option(OPT_EXPORT_BUILD_TREE "Configure the package so it is usable from the build tree.  Useful for development." OFF)
option(OPT_ARMADILLO_INT64 "Use 64-bit integers for Armadillo, Blas, and Lapack." ON)

if(NOT BUILD_SHARED_LIBS AND NOT BUILD_STATIC_LIBS)
  set (BUILD_SHARED_LIBS ON)  #Must build at least one of SHARED_ and STATIC_LIBS.  Default SHARED_
endif()

message(STATUS "OPTION: BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "OPTION: BUILD_STATIC_LIBS: ${BUILD_STATIC_LIBS}")
message(STATUS "OPTION: BUILD_TESTING: ${BUILD_TESTING}")
message(STATUS "OPTION: OPT_INSTALL_TESTING: ${OPT_INSTALL_TESTING}")
message(STATUS "OPTION: OPT_EXPORT_BUILD_TREE: ${OPT_EXPORT_BUILD_TREE}")
message(STATUS "OPTION: OPT_ARMADILLO_INT64: ${OPT_ARMADILLO_INT64}")

#Add UcommonCmakeModules git subpreo to path.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake/UncommonCMakeModules)

### Dependencies
include(AddExternalDependency)
#BacktraceException allows for exceptions that encode a backtrace for debugging
set(DEPENDENT_LIBRARY_TYPES SHARED)
if(BUILD_STATIC_LIBS)
    list(APPEND DEPENDENT_LIBRARY_TYPES STATIC)
endif()
add_external_dependency(NAME BacktraceException URL https://github.com/markjolah/BacktraceException.git ${DEPENDENT_LIBRARY_TYPES})

# Boost configure.  We require boost::math
find_package(Boost REQUIRED)

# Find Blas and Lapack using pkg-config in cross-compilation aware way.  Using local FindBLAS.cmake and FindLAPACK.cmake.
if(OPT_ARMADILLO_INT64)
    set(ARMADILLO_INT64_COMPONENT INT64) #Enable INT64 interface for armadillo, Blas and Lapack
else()
    set(ARMADILLO_INT64_COMPONENT)
endif()
find_package(BLAS REQUIRED COMPONENTS ${ARMADILLO_INT64_COMPONENT})
find_package(LAPACK REQUIRED COMPONENTS ${ARMADILLO_INT64_COMPONENT})
find_package(Armadillo REQUIRED COMPONENTS BLAS LAPACK CXX11 ${ARMADILLO_INT64_COMPONENT})
add_compile_definitions(${ARMADILLO_PRIVATE_COMPILE_DEFINITIONS})


### CFLAGS
#Configure standard CFlags and definitions for debug builds
include(ConfigureDebugBuilds)

### PackageConfig Exports
include(ExportPackageWizzard)
set(EXPORTED_FIND_MODULES FindArmadillo.cmake FindBLAS.cmake FindLAPACK.cmake MakePkgConfigTarget.cmake)
list(TRANSFORM EXPORTED_FIND_MODULES PREPEND cmake/UncommonCMakeModules/)
export_package_wizzard(PROVIDED_COMPONENTS ${ARMADILLO_INT64_COMPONENT}
                       FIND_MODULES ${EXPORTED_FIND_MODULES})

### Source directories
add_subdirectory(src)

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()
