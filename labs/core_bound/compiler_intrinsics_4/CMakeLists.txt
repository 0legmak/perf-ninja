cmake_minimum_required(VERSION 3.3)

project(lab)

set(CMAKE_CXX_STANDARD 23)

set(OUTPUT_IMAGE_PATH "${CMAKE_CURRENT_BINARY_DIR}/output.ppm")
configure_file(data_paths.h.in "${CMAKE_CURRENT_BINARY_DIR}/data_paths.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

string(REGEX MATCH "^(.*)[\\/]labs[\\/].*$" repo "${CMAKE_CURRENT_SOURCE_DIR}")
include(${CMAKE_MATCH_1}/tools/labs.cmake)
