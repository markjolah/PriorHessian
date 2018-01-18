#
# prior_hessian
# testing


set(TEST_SRCS test.cpp)
set(TEST_TARGET test_prior_hessian)
add_executable(${TEST_TARGET} ${TEST_SRCS})
target_link_libraries(${TEST_TARGET} PriorHessian)
set_target_properties(${TEST_TARGET} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(${TEST_TARGET} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
install(TARGETS ${TEST_TARGET} RUNTIME DESTINATION bin COMPONENT Runtime )

