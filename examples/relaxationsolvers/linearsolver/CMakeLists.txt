set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(linear main.cpp)

add_custom_command(TARGET linear POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:linear>/$<TARGET_FILE_NAME:linear>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:linear>
)

set_target_properties(linear PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

