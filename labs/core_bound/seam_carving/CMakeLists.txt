cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 23)

project(main LANGUAGES C CXX)

set(KERNELS_CL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl")
configure_file(data_paths.h.in "${CMAKE_CURRENT_BINARY_DIR}/data_paths.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(OpenCL CONFIG REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/labs.cmake)

target_link_libraries(lab OpenCL::OpenCL)
target_link_libraries(validate OpenCL::OpenCL)

set_target_properties(lab PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

set_target_properties(validate PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

target_compile_definitions(lab PRIVATE
  CL_HPP_MINIMUM_OPENCL_VERSION=110
  CL_HPP_TARGET_OPENCL_VERSION=110
  CL_HPP_ENABLE_EXCEPTIONS=1
)

target_compile_definitions(validate PRIVATE
  CL_HPP_MINIMUM_OPENCL_VERSION=110
  CL_HPP_TARGET_OPENCL_VERSION=110
  CL_HPP_ENABLE_EXCEPTIONS=1
)
