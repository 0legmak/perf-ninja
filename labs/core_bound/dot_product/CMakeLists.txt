cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 23)

project(main LANGUAGES CXX)

find_package(OpenCL REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/labs.cmake)

foreach(prog lab validate)
  target_link_libraries(${prog} OpenCL::OpenCL)
  set_target_properties(${prog} PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
  )
  target_compile_definitions(${prog} PRIVATE
    CL_HPP_MINIMUM_OPENCL_VERSION=110
    CL_HPP_TARGET_OPENCL_VERSION=110
    CL_HPP_ENABLE_EXCEPTIONS=1
  )
endforeach(prog)
