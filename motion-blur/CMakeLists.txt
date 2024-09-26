cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(motion_blur LANGUAGES CXX CUDA)
include(FindCUDAToolkit)

add_executable(${PROJECT_NAME} src/funcs.cc src/main.cc src/blur.cu)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
# set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "all"
    CMAKE_BUILD_TYPE Debug
)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

# set(CUDA_NVCC_FLAGS_DEBUG "-G;-lineinfo;-rdynamic;-g")

target_link_libraries(${PROJECT_NAME} argparse)
include_directories(${CMAKE_SOURCE_DIR}/deps)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

