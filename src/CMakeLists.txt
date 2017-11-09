#
# prior_hessian
#
# Parallel log-likelihood gradient and hessian calculations for composite prior distributions

file(GLOB PRIOR_HESSIAN_HDRS  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
message(STATUS "PRIOR_HESSIAN_HDRS: ${PRIOR_HESSIAN_HDRS}")
add_library(prior_hessian STATIC ${PRIOR_HESSIAN_HDRS})
set_target_properties(prior_hessian PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(prior_hessian INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> )
target_link_libraries(prior_hessian INTERFACE parallel_rng) #Parallel random number generation
set_target_properties(prior_hessian PROPERTIES PUBLIC_HEADER "${PRIOR_HESSIAN_HDRS}")
target_compile_features(prior_hessian PUBLIC cxx_std_11) #Declare C++11 required for building


get_property(_VAR TARGET prior_hessian PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "prior_hessian INTERFACE_INCLUDE_DIRECTORIES: ${_VAR}")
get_property(_VAR TARGET prior_hessian PROPERTY INTERFACE_LINK_LIBRARIES)
message(STATUS "prior_hessian INTERFACE_LINK_LIBRARIES: ${_VAR}")
get_property(_VAR TARGET prior_hessian PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "prior_hessian INCLUDE_DIRECTORIES: ${_VAR}")


install(TARGETS prior_hessian EXPORT ${EXPORT_TARGETS_NAME}
        ARCHIVE DESTINATION lib  COMPONENT Development
        PUBLIC_HEADER DESTINATION include/${PROJECT_NAME} COMPONENT Development)
