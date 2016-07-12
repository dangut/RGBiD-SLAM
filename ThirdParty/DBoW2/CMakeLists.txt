cmake_minimum_required(VERSION 2.8)
project(DBoW2)

option(BUILD_DBoW2   "Build DBoW2"            ON)

set(HDRS
  DBoW2/include/DBoW2/BowVector.h           
  DBoW2/include/DBoW2/QueryResults.h        DBoW2/include/DBoW2/TemplatedDatabase.h   DBoW2/include/DBoW2/FORB.h
  DBoW2/include/DBoW2/DBoW2.h               DBoW2/include/DBoW2/FClass.h              DBoW2/include/DBoW2/FeatureVector.h
  DBoW2/include/DBoW2/ScoringObject.h       DBoW2/include/DBoW2/TemplatedVocabulary.h 
  DUtils/include/DUtils/Random.h  DUtils/include/DUtils/Timestamp.h)
set(SRCS 
  DBoW2/src/BowVector.cpp            DBoW2/src/FORB.cpp
  DBoW2/src/FeatureVector.cpp DBoW2/src/QueryResults.cpp  DBoW2/src/ScoringObject.cpp
  DUtils/src/Random.cpp  DUtils/src/Timestamp.cpp)

find_package(OpenCV REQUIRED)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if(BUILD_DBoW2)
  include_directories(DBoW2/include/DBoW2/ DUtils/include/DUtils/ ${OpenCV_INCLUDE_DIRS})
  add_library(${PROJECT_NAME} SHARED ${SRCS})
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif(BUILD_DBoW2)

configure_file(DBoW2/src/DBoW2.cmake.in
  "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_DBoW2)
  install(DIRECTORY DBoW2/include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
