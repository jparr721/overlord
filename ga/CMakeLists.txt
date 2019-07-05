file(GLOB SOURCES "src/*.cc")
file(GLOB CUDA_SOURCES "cuda/*.cu")

add_library(ga SHARED ${SOURCES} ${CUDA_SOURCES})

target_include_directories(ga
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")
