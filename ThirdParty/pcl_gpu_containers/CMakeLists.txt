cmake_minimum_required(VERSION 2.8)
project(pcl_gpu_containers)

# Cuda
include(${CMAKE_SOURCE_DIR}/cmake_modules/pcl_find_cuda.cmake)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O2 -march=corei7 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O2 -march=corei7")

FILE(GLOB srcs src/*.cpp src/*.hpp)
FILE(GLOB incs include/*.h)
FILE(GLOB incs_impl include/impl/*.hpp)

source_group("Header Files\\impl" FILES ${incs_impl})
LIST(APPEND incs ${incs_impl})


#set(build TRUE)

if (CUDA_FOUND)
    FILE(GLOB srcs src/*.cpp src/*.hpp)
    FILE(GLOB incs include/pcl/gpu/containers/*.h)
    FILE(GLOB incs_impl include/pcl/gpu/containers/impl/*.hpp)
    
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})	
	   
	  add_library(pcl_gpu_containers SHARED ${srcs} ${incs} ${impl_incs})
	  message(pcl_gpu_containers  ${CUDA_LIBRARIES})
	  target_link_libraries(pcl_gpu_containers ${CUDA_LIBRARIES})
	  
endif()
