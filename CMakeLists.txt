cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(cerebrum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Boost 1.69 COMPONENTS unit_test_framework REQUIRED)
find_package(Eigen 3.3 REQUIRED NO_MODULE)

include_directories(${Boost_INCLUDE_DIRS})

add_subdirectory(layers)

add_executable(
  initializer_test
  test/initializers_test.cc
  )
target_link_libraries(initializer_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(initializer_test LINK_PUBLIC Eigen3::Eigen)
