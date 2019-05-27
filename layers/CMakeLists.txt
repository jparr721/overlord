file(GLOB SOURCES "src/*.cc")

add_library(layers SHARED ${SOURCES})

target_include_directories(layers PUBLIC include)
