cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)
# include(CheckLanguage)
# check_language(CUDA)

project(motion_blur LANGUAGES CXX CUDA)
include(FindCUDAToolkit)

find_package(Sail REQUIRED)

add_executable(${PROJECT_NAME} src/funcs.cc src/main.cc src/blur.cu)

set(CUDA_NVCC_FLAGS_DEBUG "-G;-lineinfo;-rdynamic;-g")

include_directories(${CMAKE_SOURCE_DIR}/deps)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

