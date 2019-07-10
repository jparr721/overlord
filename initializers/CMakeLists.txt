project(initializers)
add_library(initializers INTERFACE)

target_include_directories(${PROJECT_NAME}
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Config
        DESTINATION share/${PROJECT_NAME}/cmake
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

target_link_libraries(${PROJECT_NAME}
  INTERFACE_LINK_LIBRARIES swiss
)

install(EXPORT ${PROJECT_NAME}Config
        FILE ${PROJECT_NAME}Config.cmake
        DESTINATION lib/cmake/${PROJECT_NAME}
)
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)
