cmake_minimum_required(VERSION 4.2)
project(l0l2 CXX)

message(STATUS "Using openMP")
find_package(OpenMP REQUIRED) 
if(OpenMP_CXX_FOUND)
	link_libraries(OpenMP::OpenMP_CXX)
endif()

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
add_compile_definitions(_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS)
add_compile_options("/fp:fast")
add_compile_options("/openmp:experimental")

set(LSDIR "${PROJECT_SOURCE_DIR}/leastsquares")
set(PCADIR "${PROJECT_SOURCE_DIR}/pca")

include_directories(${LSDIR} ${PCADIR} "C:/eigen-5.0.1" "C:/thread-pool-master/include")

# Add the subdirectories
add_subdirectory(leastsquares)
add_subdirectory(pca)
add_subdirectory(examples)
add_subdirectory(python)
