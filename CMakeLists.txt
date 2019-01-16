# PriorHessian - library for fast computation of log-likelihoods, and derivatives, of multivariate priors defined with Archemedian copulas.
#
# Mark J. Olah (mjo@cs.unm DOT edu)
# Copyright 2017-2018
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
# See: LICENCE file

cmake_minimum_required( VERSION 3.9 )
project(PriorHessian VERSION 0.1 LANGUAGES CXX Fortran)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    option(BUILD_TESTING "Build testing framework" ON)
else()
    option(BUILD_TESTING "Build testing framework" OFF)
endif()

message(STATUS "OPTION: BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "OPTION: BUILD_STATIC_LIBS: ${BUILD_STATIC_LIBS}")
message(STATUS "OPTION: BUILD_TESTING: ${BUILD_TESTING}")

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake/UncommonCMakeModules)

### Dependencies
# Armadillo
find_package(Armadillo REQUIRED)
add_definitions(-DARMA_USE_CXX11)
add_definitions(-DARMA_DONT_USE_LAPACK -DARMA_DONT_USE_BLAS -DARMA_DONT_USE_OPENMP -DARMA_DONT_USE_HDF5) #disable unused dependencies
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:ARMA_PRINT_ERRORS>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<NOT:$<CONFIG:Debug>>:ARMA_NO_DEBUG>)

# Boost configure.  We require boost::math
find_package(Boost REQUIRED)

### External dependencies
include(AddExternalDependency)
#BacktraceException allows for exceptions that encode a backtrace for debugging
set(DEPENDENT_LIBRARY_TYPES SHARED)
if(BUILD_STATIC_LIBS)
    list(APPEND DEPENDENT_LIBRARY_TYPES STATIC)
endif()
add_external_dependency(NAME BacktraceException URL https://github.com/markjolah/BacktraceException.git ${DEPENDENT_LIBRARY_TYPES})

### CFLAGS
#Configure standard CFlags and definitions for debug builds
include(ConfigureDebugBuilds)

### PackageConfig Exports
include(ExportPackageWizzard)
export_package_wizzard()

### Source directories
add_subdirectory(src)

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()
