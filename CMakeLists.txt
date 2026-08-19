cmake_minimum_required(VERSION 3.24)
project(nyangrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# the tiled cpu matmul leans on vectorisation, so keep the optimiser flags
if(MSVC)
  set(CMAKE_CXX_FLAGS "/O2 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-O2 -march=native ${CMAKE_CXX_FLAGS}")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

set(BACKEND_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nyangrad)

pybind11_add_module(_cpu_backend csrc/cpu_backend.cc)
set_target_properties(_cpu_backend PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${BACKEND_OUTPUT_DIR})

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "nvcc found, building the cuda backend")
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  pybind11_add_module(_cuda_backend csrc/cuda_backend.cu)
  set_target_properties(_cuda_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${BACKEND_OUTPUT_DIR}
    CUDA_VISIBILITY_PRESET hidden)
else()
  message(STATUS "no nvcc on PATH, skipping the cuda backend")
endif()
