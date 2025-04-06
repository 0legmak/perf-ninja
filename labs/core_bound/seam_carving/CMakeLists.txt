cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 23)

project(seam_carving LANGUAGES CXX)

set(KERNELS_CL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl")
set(IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")
configure_file(data_paths.h.in "${CMAKE_CURRENT_BINARY_DIR}/data_paths.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(OpenCL REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/labs.cmake)

foreach(prog lab validate)
  target_link_libraries(${prog} OpenCL::OpenCL)
  target_compile_definitions(${prog} PRIVATE
    CL_HPP_MINIMUM_OPENCL_VERSION=110
    CL_HPP_TARGET_OPENCL_VERSION=110
    CL_HPP_ENABLE_EXCEPTIONS=1
  )
endforeach(prog)
