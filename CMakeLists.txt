cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(cerebrum LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Boost 1.69 COMPONENTS unit_test_framework REQUIRED)
find_package(Armadillo 9.3 REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${ARMADILLO_INCLUDE_DIRS})

