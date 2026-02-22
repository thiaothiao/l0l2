cmake_minimum_required(VERSION 4.2)
project(l0l2 CXX)

message(STATUS "Using openMP")
find_package(OpenMP REQUIRED) 
if(OpenMP_CXX_FOUND)
	link_libraries(OpenMP::OpenMP_CXX)
endif()

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
add_compile_definitions(_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS)

set(LSDIR "${PROJECT_SOURCE_DIR}/leastsquares")
set(PCADIR "${PROJECT_SOURCE_DIR}/pca")
message(STATUS "Using Eigen3")
message(STATUS "Using thread-pool")
include_directories(${LSDIR} ${PCADIR} "$ENV{EIGEN3_INCLUDE_DIR}" "$ENV{THREAD_POOL_INCLUDE_DIR}")

# Add the subdirectories
add_subdirectory(leastsquares)
add_subdirectory(pca)
add_subdirectory(examples)
add_subdirectory(python)
