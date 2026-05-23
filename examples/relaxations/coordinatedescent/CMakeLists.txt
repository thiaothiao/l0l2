set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(coordinatedescent main.cpp)

add_custom_command(TARGET coordinatedescent POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:coordinatedescent>/$<TARGET_FILE_NAME:coordinatedescent>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:coordinatedescent>
)

set_target_properties(coordinatedescent PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

