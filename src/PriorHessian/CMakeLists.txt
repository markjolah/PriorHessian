# PriorHessian
#
# Header-only library
# 
# Fast log-likelihood gradient and hessian calculations for composite prior distributions

file(GLOB HDRS  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
add_library(PriorHessian INTERFACE)
#target_include_directories(PriorHessian INTERFACE $<INSTALL_INTERFACE:include>)
target_include_directories(PriorHessian INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>)
target_link_libraries(PriorHessian INTERFACE BacktraceException)
target_compile_features(PriorHessian INTERFACE cxx_std_14) #C++14 required for variadic templates support.


get_property(_VAR TARGET ${PROJECT_NAME} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "${PROJECT_NAME} INTERFACE_INCLUDE_DIRECTORIES: ${_VAR}")
get_property(_VAR TARGET ${PROJECT_NAME} PROPERTY INTERFACE_LINK_LIBRARIES)
message(STATUS "${PROJECT_NAME} INTERFACE_LINK_LIBRARIES: ${_VAR}")
get_property(_VAR TARGET ${PROJECT_NAME} PROPERTY INTERFACE_COMPILE_FEATURES)
message(STATUS "${PROJECT_NAME} INTERFACE_COMPILE_FEATURES: ${_VAR}")

install(TARGETS PriorHessian EXPORT ${EXPORT_TARGETS_NAME})
install(FILES ${HDRS} DESTINATION include/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rng DESTINATION include/${PROJECT_NAME})

