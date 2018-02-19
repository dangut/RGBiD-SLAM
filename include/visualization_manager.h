/**
* This file is part of RGBID-SLAM.
*
* Copyright (C) 2015 Daniel Gutiérrez Gómez <danielgg at unizar dot es> (Universidad de Zaragoza)
*
* RGBID-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* RGBID-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with RGBID-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef VISUALIZATIONMANAGER_HPP_
#define VISUALIZATIONMANAGER_HPP_


#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <deque>
#include <boost/thread.hpp>

//#include "internal.h"
#include "types.h"

//#include <vtkSmartPointer.h>
//#include <vtkImageViewer2.h>
//#include <vtkRenderWindow.h>
//#include <vtkRenderWindowInteractor.h>
//#include <vtkRenderer.h>

#include <vtkImageData.h>
#include <vtkImageMapper.h> // Note: this is a 2D mapper (cf. vtkImageActor which is 3D)
#include <vtkActor2D.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkImageFlip.h>

//////////////////////////////////////////



namespace RGBID_SLAM
{ 
  struct ImageView
  {
    ImageView();
    
    void 
    setViewer(int pos_x, int pos_y, std::string window_name= "Image viewer", float min_val = 0.f, float max_val = 255.f);
    
    void
    showRGB (std::vector<PixelRGB>& rgb24, int cols, int rows);
    
    void
    showDepth (std::vector<unsigned short>& depth, int cols, int rows);
    
    void
    showFloat (std::vector<float>& intensity, int cols, int rows);
    
    
    
    private:
      void
      addRGB (std::vector<PixelRGB>& rgb24, int cols, int rows);
    
      template<class T> void
      transformToRGB(std::vector<T>&in, std::vector<PixelRGB>& rgb24)
      {
        rgb24.resize(in.size());
        
        for (int i=0; i<in.size(); i++){
          float value = static_cast<float> (in[i]);
          float value_norm = (value - min_val_)/ (max_val_-min_val_);
          value_norm = std::max(0.f, std::min(1.f, value_norm));
          
          PixelRGB rgb_val;
          rgb_val.r = rgb_val.b = rgb_val.g = static_cast<unsigned char>(255.f*value_norm);
          rgb24[i] = rgb_val;        
        }
      }
    //pcl::visualization::ImageViewer viewerRGB_; 
    //vtkSmartPointer<vtkImageViewer2> viewer_;
    //vtkSmartPointer<vtkRenderWindowInteractor> interactor_;
    
    vtkSmartPointer<vtkImageMapper> mapper_; 
    vtkSmartPointer<vtkActor2D> image_;
    vtkSmartPointer<vtkRenderer> renderer_ ;
    vtkSmartPointer<vtkRenderWindow> window_ ;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor_ ;
    vtkSmartPointer<vtkImageFlip> flipper_;
    
    float min_val_;
    float max_val_;
  };
    

  struct FloatView
  {
    FloatView();
    
    void 
    setViewer(int pos_x, int pos_y, float min_val = 0.f, float max_val = 255.f,  std::string window_name= "Float stream");
    
    void
    showFloat (const PtrStepSz<const float>& intensity);
    
    int min_val_;
    int max_val_;
    
    pcl::visualization::ImageViewer viewerFloat_; 
  };

  //////////////////////////////////////////
  struct DepthView
  {
    DepthView();
    
    void
    setViewer(int pos_x, int pos_y, int min_val=0, int max_val=10000,  std::string window_name= "Kinect Depth stream");
    
    void
    showDepth (const PtrStepSz<const unsigned short>& depth) ;
    
    float min_val_;
    float max_val_;
    
    pcl::visualization::ImageViewer viewerDepth_; 
  };


  // View the camera
  struct CameraView
  {
    public:
    
      CameraView();
      
      void
      setViewer(int pos_x, int pos_y, std::string window_name= "3D world");
      
      void
      setViewerPose (const Eigen::Affine3d& viewer_pose);
      
      void updateCamera(const Eigen::Affine3d &new_pose);
      
      void updateKeyframe(const Eigen::Affine3d &keyframe_pose, int kf_id);
    
      void updatePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_point_cloud, int cloud_id, Eigen::Affine3d pose=Eigen::Affine3d::Identity());
      
      void updateTrajectory(pcl::PointCloud<pcl::PointXYZ>::Ptr& trajectory);
      
      void spinOnce(int t=1);
      
      void removeShapes();
      
    private:
      
      void drawMotionUncertainty(const Eigen::Matrix<float,6,6> cov_mat, const Eigen::Affine3f& pose, double r, double g, double b);
      
      void drawCamera (const Eigen::Affine3f& pose, double r, double g, double b, double s = 1.0);
      
      void drawKeyframe (const Eigen::Affine3f& pose, int kf_id, double r, double g, double b, double s = 1.0);
      
      bool removeCamera ();
      
      bool removeKeyframe (int kf_id);
      
      Eigen::Affine3f viewer_pose_;
      Eigen::Affine3f camera_pose_;
      Eigen::Affine3f camera_pose_prev_;
      Eigen::Matrix<float,6,6> cov_mat_;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr viewer_point_cloud_;
      
      std::string name_;
      pcl::visualization::PCLVisualizer cloud_viewer_;
      int cloud_number_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  };


  class VisualizationManager
  {
    public:
      VisualizationManager();
      ~VisualizationManager();
    
      boost::mutex mutex_, visodo_mutex_;
      
      Eigen::Affine3d new_pose_;
      Eigen::Affine3d last_KF_pose_;    
      int kf_id_;
      int new_pc_id_;
      
      BufferWrapperEigen<Eigen::Affine3d> buffer_poses_;
      BufferWrapperEigen<Eigen::Affine3d> buffer_odo_;
      
      std::vector<PixelRGB> scene_view_;
      std::vector<PixelRGB> segmentation_view_;
      std::vector<PixelRGB> negentropy_view_;
      std::vector<float> intensity_view_;
      std::vector<float> depthinv_view_;
      
      IdToPointCloudMap id2pointcloud_map_;
      IdToPoseMap id2pose_map_;
      pcl::PointCloud<pcl::PointXYZ>::Ptr trajectory_;
      
      void copyTrajectory(const std::vector<Pose> &cam_poses);        
      void getPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr whole_point_cloud);        
    
      void start();
      void refresh();
      void stop();
      
      void operator() ();
      
      VisodoTrackerPtr visodo_ptr_;
      KeyframeManagerPtr keyframe_manager_ptr_; 
      
    private:
      
      CameraView camera_viewer_;
      //FloatView intensity_viewer_;
      ImageView intensity_viewer_;
      ImageView scene_viewer_;
      ImageView negentropy_viewer_;
      //ImageView segmentation_viewer_;
      
      boost::shared_ptr<boost::thread> drawing_thread_;
      bool stop_;
      int cols_;
      int rows_;
      
      bool redraw_pointcloud_;
      bool redraw_trajectory_;
      bool redraw_camera_;
      bool redraw_all_pointclouds_;
      bool redraw_scene_view_;
      bool redraw_keyframe_;
      
    public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
     
  };
}

  
#endif
 
