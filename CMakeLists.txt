cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(VISODO_RGBD)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules/")
message(${CMAKE_MODULE_PATH})



SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unknown-pragmas -fno-strict-aliasing -Wno-format-extra-args -Wno-sign-compare -Wno-invalid-offsetof -Wno-conversion -march=corei7 -O2 -pipe")
  
find_package(catkin REQUIRED COMPONENTS
  roscpp  
  openni2_launch
  sensor_msgs  
  pcl_ros
  message_filters
  tf
  tf_conversions
  rosconsole
  rostime
)

include_directories(${catkin_INCLUDE_DIRS})
 
 
#set(CMAKE_PREFIX_PATH "/usr/local/")
find_package(PCL 1.7  REQUIRED COMPONENTS common io visualization filters)
message(${PCL_FOUND})
include_directories(BEFORE ${PCL_INCLUDE_DIRS})
link_directories( ${PCL_LIBRARY_DIRS})
add_definitions( ${PCL_DEFINITIONS})

# Required boost modules
set(BOOST_REQUIRED_MODULES system filesystem thread date_time iostreams serialization math_c99)
find_package(Boost 1.40.0 REQUIRED COMPONENTS ${BOOST_REQUIRED_MODULES})
if(Boost_FOUND)
  set(BOOST_FOUND TRUE)
  # Obtain diagnostic information about Boost's automatic linking outputted 
  # during compilation time.
  add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
  include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif(Boost_FOUND)


# Eigen (required)
find_package(Eigen 3.1 REQUIRED)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})
add_definitions(-DEIGEN_USE_NEW_STDVECTOR
                -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)
                
# OpenNI
find_package(OpenNI REQUIRED)
include_directories(SYSTEM ${OPENNI_INCLUDE_DIRS})

#set(build FALSE)

# Search for OpenCV 
find_package(OpenCV QUIET) 
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
add_definitions(-DHAVE_OPENCV) 

add_definitions("-DBOOST_DISABLE_ASSERTS -DEIGEN_NO_DEBUG")

# Cuda
include(${CMAKE_SOURCE_DIR}/cmake_modules/pcl_find_cuda.cmake)

#DBoW2
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/DBoW2/DBoW2/include/ ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/DBoW2/DUtils/include/DUtils/)	
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/DBoW2/lib)
message(${OpenCV_INCLUDE_DIRS})
#pcl_gpu_containers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/pcl_gpu_containers/include)	
#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/pcl_gpu_containers/lib)

#g2o directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/g2o-lite ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/g2o-lite/build/ )	
set(MY_G2O_LIBDIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/g2o-lite/lib")
SET(G2O_LIBS ${MY_G2O_LIBDIR}/libg2o_core.so  ${MY_G2O_LIBDIR}/libg2o_stuff.so ${MY_G2O_LIBDIR}/libg2o_types_pose_graph.so  ${MY_G2O_LIBDIR}/libg2o_solver_eigen.so)

string(REPLACE "-Wold-style-cast" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-Wno-invalid-offsetof" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function")
if (GCC_VERSION VERSION_GREATER 4.3)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
endif()
    
    
set(build TRUE)
if (build)
        get_directory_property(_dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(_vtk_definitions)
    foreach(_item ${_dir_defs})
        if(_item MATCHES "vtk*")
            list(APPEND _vtk_definitions -D${_item})
        endif()
    endforeach()
message(${_vtk_definitions})
remove_definitions(${_vtk_definitions})

	FILE(GLOB incs include/*.h*)
	FILE(GLOB srcs src/*.cpp src/*.h*)
	FILE(GLOB cuda src/cuda/*.cu src/cuda/*.h*)
			
	source_group("Source Files\\cuda" FILES ${cuda} )    
	source_group("Source Files" FILES ${srcs} )		
	
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src ${CUDA_INCLUDE_DIRS})	
	
	if (UNIX OR APPLE)
		set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;--default-stream per-thread")			
	endif()
	
	if(NOT UNIX OR APPLE)
		add_definitions(-DPCLAPI_EXPORTS)
	endif()
	
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false;--ptxas-options=-v ")		
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false ")		
#  message (${CUDA_NVCC_FLAGS})
	CUDA_COMPILE(cuda_objs ${cuda})	
	
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -o3 -g")	
	 
	add_library(RGBID_SLAM ${srcs} ${incs} ${impl_incs} ${cuda} ${cuda_objs})
#	message(RGBID_SLAM ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CUDA_LIBRARIES} DBoW2 ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/pcl_gpu_containers/lib/libpcl_gpu_containers.so  ${G2O_LIBS})
	target_link_libraries(RGBID_SLAM  ${Boost_LIBRARIES} ${CUDA_LIBRARIES} ${PCL_LIBRARIES} DBoW2 ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/pcl_gpu_containers/lib/libpcl_gpu_containers.so  ${G2O_LIBS})
  if(OpenCV_FOUND) 
     target_link_libraries(RGBID_SLAM ${OpenCV_LIBS}) 
	endif() 
			
		
	add_subdirectory(tools)
endif()


