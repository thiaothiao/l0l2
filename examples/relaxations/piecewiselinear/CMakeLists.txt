set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(piecewiselinear main.cpp)

add_custom_command(TARGET piecewiselinear POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:piecewiselinear>/$<TARGET_FILE_NAME:piecewiselinear>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:piecewiselinear>
)

set_target_properties(piecewiselinear PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

