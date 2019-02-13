# PriorHessian/test/CMakeLists.txt

find_package(GTest REQUIRED)

set(TEST_TARGET test${PROJECT_NAME})
file(GLOB GTEST_SRCS test_*.cpp)

add_executable(${TEST_TARGET} ${GTEST_SRCS})
target_link_libraries(${TEST_TARGET} PUBLIC ${PROJECT_NAME}::${PROJECT_NAME})
target_link_libraries(${TEST_TARGET} PUBLIC GTest::GTest)
set_target_properties(${TEST_TARGET} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
add_test(NAME GTest COMMAND ${TEST_TARGET})
if(WIN32 AND CMAKE_CROSSCOMPILING)
    #Need the -O1 optimization inlining to reduce the number of symbols
    get_target_property(_opts ${TEST_TARGET} COMPILE_OPTIONS)
    string(REPLACE "-O0" "-O1" _opts "${_opts}")
    set_target_properties(${TEST_TARGET} PROPERTIES COMPILE_OPTIONS ${_opts})
endif()
if(OPT_INSTALL_TESTING)
    if(WIN32)
        set(TESTING_INSTALL_DESTINATION bin)
    elseif(UNIX)
        set(TESTING_INSTALL_DESTINATION lib/${PROJECT_NAME}/test)
        set_target_properties(${TEST_TARGET} PROPERTIES INSTALL_RPATH "\$ORIGIN/../..")
    endif()
    install(TARGETS ${TEST_TARGET} RUNTIME DESTINATION ${TESTING_INSTALL_DESTINATION} COMPONENT Testing)
endif()
