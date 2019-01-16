cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project(mainP3P)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_64_BIT_DEVICE_CODE ON)

include_directories( ${OpenCV_INCLUDE_DIRS})


cuda_add_executable(mainP3P mainP3P.cpp P3PSolver.cpp RANSAC.cpp
                    kernelP3P.cu kernelP3P.cuh)

target_link_libraries(mainP3P ${OpenCV_LIBS})
