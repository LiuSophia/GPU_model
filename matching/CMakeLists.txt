cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project(main)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_64_BIT_DEVICE_CODE ON)

include_directories( ${OpenCV_INCLUDE_DIRS})

cuda_add_executable(main main.cpp 
                    kernel.cu kernel.cuh)

target_link_libraries(main ${OpenCV_LIBS})
