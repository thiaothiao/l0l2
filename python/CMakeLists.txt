find_package(Python3 COMPONENTS Interpreter Development REQUIRED) # the Development component is required, else we get an error: https://github.com/pybind/pybind11/issues/3996

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})

message(STATUS "Using pybind11 found at: ${pybind11_DIR}")
find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(l0l2 SHARED pybindwrapper.cpp)

# Needed for importing the "l0l2" module from the build directory
set(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
if(NOT EXISTS ${INIT_PY})
    file(WRITE ${INIT_PY} "")
endif()

add_custom_command(TARGET l0l2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:l0l2>/$<TARGET_FILE_NAME:l0l2>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:l0l2>
)

set_target_properties(l0l2 PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)