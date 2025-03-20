cmake_minimum_required(VERSION 3.5)

project(lab)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

set(OUTPUT_IMAGE_PATH "${CMAKE_CURRENT_BINARY_DIR}/output.ppm")
configure_file(data_paths.h.in "${CMAKE_CURRENT_BINARY_DIR}/data_paths.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/labs.cmake)
