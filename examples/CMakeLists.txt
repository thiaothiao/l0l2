add_executable(leastsquares leastsquares.cpp simu.hpp)

add_custom_command(TARGET leastsquares POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:leastsquares>/$<TARGET_FILE_NAME:leastsquares>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:leastsquares>
)

add_executable(pca pca.cpp simu.hpp)
if (MSVC)
    target_compile_options(pca PRIVATE /bigobj)
endif()

add_custom_command(TARGET pca POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:pca>/$<TARGET_FILE_NAME:pca>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:pca>
)

set_target_properties(leastsquares pca PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
