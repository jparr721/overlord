file(GLOB SOURCES "src/*.cc")

add_library(initializers SHARED ${SOURCES})

target_include_directories(initializers PUBLIC include)
