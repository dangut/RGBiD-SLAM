ADD_LIBRARY(types_pose_graph ${G2O_LIB_TYPE}
  types_six_dof_pose.h
  types_six_dof_pose.cpp
  g2o_types_pose_graph_api.h
)

SET_TARGET_PROPERTIES(types_pose_graph PROPERTIES OUTPUT_NAME ${LIB_PREFIX}types_pose_graph)

TARGET_LINK_LIBRARIES(types_pose_graph core)

INSTALL(TARGETS types_pose_graph
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

INSTALL(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/g2o/types/pose_graph)
