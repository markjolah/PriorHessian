# PriorHessian
#
# Header-only library
# 
# Fast log-likelihood gradient and hessian calculations for composite prior distributions


set(PUBLIC_HEADER_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../include)


add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PUBLIC_HEADER_SRC_DIR}> 
                                                     $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} INTERFACE BacktraceException::BacktraceException) #For BacktraceException library
target_link_libraries(${PROJECT_NAME} INTERFACE ${ARMADILLO_LIBRARIES}) #For Boost header-only libraries
target_link_libraries(${PROJECT_NAME} INTERFACE Boost::boost) #For Boost header-only libraries
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14) #C++14 required for variadic templates support.
target_compile_options(${PROJECT_NAME} INTERFACE $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-parameter>)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets)
install(DIRECTORY ${PUBLIC_HEADER_SRC_DIR}/ DESTINATION include COMPONENT Development)

