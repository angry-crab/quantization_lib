cmake_minimum_required(VERSION 2.8.7)
project(centerpoint)
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

if(${arch}     STREQUAL  "aarch64")
  set(TENSORRT_ROOT         /usr/lib/aarch64-linux-gnu)
  # set(CUDA_TOOLKIT_ROOT_DIR /root/.kiwi/lib/cuda-11.8)
elseif(${arch} STREQUAL "x86_64")
  set(TENSORRT_ROOT         /usr/lib/x86_64-linux-gnu)
  # set(CUDA_TOOLKIT_ROOT_DIR /root/.kiwi/lib/cuda-11.8)
endif()

# set(TENSORRT_ROOT         /usr/lib/x86_64-linux-gnu)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-deprecated-declarations -O3")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_87,code=compute_87)
set(TENSORRT_INCLUDE ${TENSORRT_ROOT})
set(TENSORRT_LIB     ${TENSORRT_ROOT})

find_package(Eigen3 REQUIRED)
message("Eigen3")
message(STATUS "${EIGEN3_INCLUDE_DIR}")

find_library(NVONNXPARSER nvonnxparser)
if(NVONNXPARSER)
  message("NVONNXPARSER ok")
  message(STATUS "${NVONNXPARSER}")
endif()
find_package(CUDA REQUIRED)
# enable_language(CUDA)
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE}
  ${EIGEN3_INCLUDE_DIR}
  include
)

message("CUDA")
message(STATUS "${CUDA_INCLUDE_DIRS}")

link_directories(
  ${TENSORRT_LIB}
  ${NVONNXPARSER}
)

file(GLOB_RECURSE SOURCE_FILES 
  src/*.cu
  src/*.cpp
)

cuda_add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})

target_link_LIBraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
    Eigen3::Eigen
)

# add_executable(main main.cpp src/voxel_generator.cpp src/utils.cpp)