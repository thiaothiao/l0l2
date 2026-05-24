set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(branchandbound main.cpp)

add_custom_command(TARGET branchandbound POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:branchandbound>/$<TARGET_FILE_NAME:branchandbound>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:branchandbound>
)

set_target_properties(branchandbound PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
