
if(NOT VTK_FOUND)
  set(DEFAULT FALSE)
  set(REASON "VTK was not found.")
else(NOT VTK_FOUND)
  set(DEFAULT TRUE)
  set(REASON)
  set(VTK_USE_FILE ${VTK_USE_FILE} CACHE INTERNAL "VTK_USE_FILE")
  include (${VTK_USE_FILE})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
endif(NOT VTK_FOUND)

FILE(GLOB hdrs "*.h*")
include_directories(${VTK_INCLUDE_DIRS})


	
## VISODO
set(the_target RGBID_SLAMapp)
set(srcs RGBID_SLAMapp.cpp evaluation.cpp)

source_group("Source Files" FILES ${srcs} )

add_executable(${the_target} ${srcs} ${hdrs})
target_link_libraries(${the_target} ${PCL_LIBRARIES} ${OPENNI_LIBRARIES} ${Boost_LIBRARIES} ${CUDA_LIBRARIES} RGBID_SLAM)  
target_link_libraries(${the_target} ${OpenCV_LIBS} ${catkin_LIBRARIES} ncurses) 	



#OPENNI_GRABBER 
set(the_target openni_grabber)
set(srcs openni_grabber_viewer_save_images_buffered.cpp)

source_group("Source Files" FILES ${srcs} )

add_executable(${the_target} ${srcs} ${hdrs})
target_link_libraries(${the_target} ${PCL_LIBRARIES} ${OPENNI_LIBRARIES} ${Boost_LIBRARIES})  
target_link_libraries(${the_target} ${OpenCV_LIBS} ${catkin_LIBRARIES} ) 	



#OPENNI_GRABBER 
set(the_target openni_rgb_depth_ir_shot)
set(srcs openni_rgb_depth_ir_shot.cpp)

source_group("Source Files" FILES ${srcs} )

add_executable(${the_target} ${srcs} ${hdrs})
target_link_libraries(${the_target} ${PCL_LIBRARIES} ${OPENNI_LIBRARIES} ${Boost_LIBRARIES})  
target_link_libraries(${the_target} ${OpenCV_LIBS} ${catkin_LIBRARIES} ) 	


