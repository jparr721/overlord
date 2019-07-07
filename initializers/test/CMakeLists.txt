project(initializers_tests)

set(TEST_SOURCES
  initializers_test.cc
)

add_executable(${PROJECT_NAME}_test_main ${TEST_SOURCES})
target_link_libraries(
  ${PROJECT_NAME}_test_main
  ${Boost_LIBRARIES}
  Eigen3::Eigen
  initializers
  layers
)
