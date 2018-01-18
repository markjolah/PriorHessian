# PriorHessian - library for fast computation of log-likelihoods, and derivatives, of multivariate priors defined with Archemedian copulas.
#
# Mark J. Olah (mjo@cs.unm DOT edu)
# Copyright 2017
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
# See: LICENCE file

cmake_minimum_required( VERSION 3.8 )
project(PriorHessian VERSION 0.1 LANGUAGES CXX)
cmake_policy(SET CMP0054 NEW) #Don't derrefernce in if() statements
cmake_policy(SET CMP0022 NEW) #LINK_INTERFACE_LIBRARIES fix

#Add to modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules 
                                           ${CMAKE_CURRENT_LIST_DIR}/cmake/Functions)

# Armadillo
find_package(Armadillo REQUIRED)
add_definitions(-DARMA_USE_CXX11 -DARMA_DONT_USE_WRAPPER -DARMA_BLAS_LONG)
add_definitions(-DARMA_DONT_USE_LAPACK) #Should not be needed
add_definitions(-DARMA_DONT_USE_BLAS) #Should not be needed
add_definitions(-DARMA_DONT_USE_OPENMP) #Want to explicitly use openMP when required
add_definitions(-DARMA_DONT_USE_HDF5) #Should not be needed
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    add_definitions(-DARMA_PRINT_ERRORS)
endif()

#Boost configure
find_package(Boost)

#External dependencies
include(AddExternalDependency)
#BacktraceException allows for exceptions that encode a backtrace for debugging
set(BacktraceExceptionURL https://github.com/markjolah/BacktraceException.git)
AddExternalDependency(BacktraceException ${BacktraceExceptionURL} SHARED)

# Compiler Definitions
if (WIN32)
    add_definitions( -DWIN32 )
elseif(UNIX AND NOT APPLE)
    add_definitions( -DLINUX )
endif()

# CFLAGS ##
add_compile_options(-W -Wall -Wextra -Werror -Wno-unused-parameter)
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    add_definitions(-DDEBUG)
elseif()
    add_definitions(-DNDEBUG)
endif()
set(CMAKE_DEBUG_POSTFIX ".debug" CACHE STRING "Debug file extension")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-ggdb -O3")

include(PackageConfig) #Configure Package-Config and associated files for cmake exports

add_subdirectory(src/PriorHessian)
add_subdirectory(src/test)

