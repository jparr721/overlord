file(GLOB SOURCES "src/*.cc")

add_library(activations SHARED ${SOURCES})

target_include_directories(activations PUBLIC include)
