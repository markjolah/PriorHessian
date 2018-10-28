#
# test/CMakeLists.txt
# CTest integration of googletest

find_package(GTest REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

remove_definitions(-DARMA_DONT_USE_LAPACK)
remove_definitions(-DARMA_DONT_USE_BLAS)
add_definitions(-DARMA_USE_LAPACK)
add_definitions(-DARMA_USE_BLAS)
set(GTEST_EXE test${PROJECT_NAME})
file(GLOB GTEST_SRCS test_*.cpp)
add_executable(${GTEST_EXE} ${GTEST_SRCS})
target_link_libraries(${GTEST_EXE} ${PROJECT_NAME}::${PROJECT_NAME}) 
target_link_libraries(${GTEST_EXE} GTest::GTest GTest::Main)
# target_link_libraries(${GTEST_EXE} ${ARMADILLO_LIBRARIES})

add_executable(debug_bvn debug_bvn.cpp)
target_link_libraries(debug_bvn ${PROJECT_NAME}::${PROJECT_NAME}) 

add_test(NAME GTest COMMAND ${GTEST_EXE})
