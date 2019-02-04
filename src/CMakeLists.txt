# PriorHessian/src
# Main CMake for PriorHessian libraries

file(GLOB SRCS *.cpp)  #Gather all .cpp sources
list(APPEND SRCS mvndstpack.f) #Add fortran sources

include(AddSharedStaticLibraries)
add_shared_static_libraries(SOURCES ${SRCS}
                            PUBLIC_HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/../include
                            PUBLIC_DEBUG_HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/../include/debug
                            RETURN_TARGETS lib_targets)

foreach(target IN LISTS lib_targets)
    target_compile_features(${target} PUBLIC cxx_std_14) #C++14 required for variadic templates support.
    target_compile_options(${target} INTERFACE $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-parameter>)
    target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADER_SRC_DIR}> $<INSTALL_INTERFACE:include>)
    if(OPT_ARMADILLO_INT64)
        target_link_libraries(${target} INTERFACE LAPACK::LapackInt64 BLAS::BlasInt64)
    else()
        target_link_libraries(${target} INTERFACE LAPACK::Lapack BLAS::Blas)
    endif()
    target_link_libraries(${target} INTERFACE Armadillo::Armadillo)
    target_link_libraries(${target} PUBLIC Boost::boost) #For Boost header-only libraries
endforeach()
