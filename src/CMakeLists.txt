# PriorHessian/src
# Main CMake for PriorHessian libraries

file(GLOB SRCS *.cpp)  #Gather all .cpp sources
list(APPEND SRCS mvndstpack.f) #Add fortran sources

include(AddSharedStaticLibraries)
# add_shared_static_libraries()
# * Add shared and static library targets to project namespace
# * Sets target_include_directories
# * Sets target_compile_features
# * Installs target libraries and headers
# * returns list: lib_targets - of created targets for adjusting additional target properties
add_shared_static_libraries(SOURCES ${SRCS}
                            COMPILE_FEATURES cxx_std_14
                            PUBLIC_HEADER_DIR ${CMAKE_SOURCE_DIR}/include
                            PUBLIC_DEBUG_HEADER_DIR ${CMAKE_SOURCE_DIR}/include/debug
                            RETURN_TARGETS lib_targets)

#Custom target settings for each lib_target from add_shared_static_libraries
foreach(target IN LISTS lib_targets)
    target_compile_options(${target} INTERFACE $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-parameter>
                                                   $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-but-set-parameter>
                                                   $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-but-set-variable> )
    #Boost extendend precision math has problems with older GCC compilers
    target_compile_options(${target} PUBLIC $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:${CMAKE_CXX_COMPILER_VERSION},7>,$<COMPILE_LANGUAGE:CXX>>:-fext-numeric-literals>)

    if(OPT_BLAS_INT64)
        target_link_libraries(${target} INTERFACE LAPACK::LapackInt64 BLAS::BlasInt64)
    else()
        target_link_libraries(${target} INTERFACE LAPACK::Lapack BLAS::Blas)
    endif()
    target_link_libraries(${target} PUBLIC Armadillo::Armadillo)
    target_link_libraries(${target} PUBLIC Boost::boost) #For Boost header-only libraries
endforeach()
