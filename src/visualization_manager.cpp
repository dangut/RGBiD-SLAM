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

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include "visualization_manager.h"
#include "util_funcs.h"
#include "visodo.h"
#include "keyframe_manager.h"


#include <Eigen/Core>


static boost::mutex display_mutex_;
//////////////////////////////////////////
RGBID_SLAM::ImageView::ImageView()
{
  boost::mutex::scoped_lock lock(display_mutex_);
  viewerRGB_.setWindowTitle ("RGB stream");
  //viewerRGB_.setPosition (0, 0);
}
  
void
RGBID_SLAM::ImageView::setViewer(int pos_x, int pos_y, std::string window_name)
{
  boost::mutex::scoped_lock lock(display_mutex_);
  viewerRGB_.setWindowTitle (window_name);
  viewerRGB_.setPosition (pos_x, pos_y);
}
  
void
RGBID_SLAM::ImageView::showRGB (const PtrStepSz<const PixelRGB>& rgb24) 
{ 
  boost::mutex::scoped_lock lock(display_mutex_);
  viewerRGB_.showRGBImage ((unsigned char*)&rgb24.data[0], rgb24.cols, rgb24.rows);
}

RGBID_SLAM::FloatView::FloatView()
{  
  min_val_ = 0.f;
  max_val_ = 255.f;
}

void 
RGBID_SLAM::FloatView::setViewer(int pos_x, int pos_y, float min_val, float max_val,  std::string window_name)
{
  boost::mutex::scoped_lock lock(display_mutex_);
  viewerFloat_.setWindowTitle (window_name);
  viewerFloat_.setPosition (pos_x, pos_y);
  min_val_ = min_val;
  max_val_ = max_val;
}

void
RGBID_SLAM::FloatView::showFloat (const PtrStepSz<const float>& intensity) 
{ 
  boost::mutex::scoped_lock lock(display_mutex_);
  viewerFloat_.showFloatImage ((float*)&intensity.data[0], intensity.cols, intensity.rows, min_val_, max_val_, true);
}
  
//////////////////////////////////////////
RGBID_SLAM::DepthView::DepthView()
{
  boost::mutex::scoped_lock lock(display_mutex_);
  viewerDepth_.setWindowTitle ("Kinect Depth stream");
  //viewerDepth_.setPosition (0, 0);
  min_val_ = 0;
  max_val_ = 10000;
}

void
RGBID_SLAM::DepthView::setViewer(int pos_x, int pos_y, int min_val, int max_val,  std::string window_name)
{
  boost::mutex::scoped_lock lock(display_mutex_);
  viewerDepth_.setWindowTitle (window_name);
  viewerDepth_.setPosition (pos_x, pos_y);
  min_val_ = min_val;
  max_val_ = max_val;
}

void
RGBID_SLAM::DepthView::showDepth (const PtrStepSz<const unsigned short>& depth) 
{ 
  boost::mutex::scoped_lock lock(display_mutex_);
  viewerDepth_.showShortImage ((unsigned short*)&depth.data[0], depth.cols, depth.rows, min_val_, max_val_);
}

// View the camera
RGBID_SLAM::CameraView::CameraView() : cloud_viewer_ ("Camera Viewer")
{
  boost::mutex::scoped_lock lock(display_mutex_);    
  viewer_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);  
  
  cloud_viewer_.setBackgroundColor (0, 0, 0);
  cloud_viewer_.addCoordinateSystem (1.0);
  cloud_viewer_.initCameraParameters ();
  cloud_viewer_.setPosition (0, 0);
  cloud_viewer_.setSize (3*640, 480);
  cloud_viewer_.setCameraClipDistances (0.01, 10.01);  
  cov_mat_ = Eigen::Matrix<float, 6, 6>::Identity();
}
  
void
RGBID_SLAM::CameraView::setViewer(int pos_x, int pos_y, std::string window_name)
{
  boost::mutex::scoped_lock lock(display_mutex_);
  viewer_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>); 
  
  cloud_viewer_.setBackgroundColor (0, 0, 0);
  cloud_viewer_.addCoordinateSystem (1.0);
  cloud_viewer_.initCameraParameters ();
  cloud_viewer_.setPosition (0, 0);
  cloud_viewer_.setSize (3*640, 480);
  cloud_viewer_.setCameraClipDistances (0.01, 10.01);  
  cov_mat_ = Eigen::Matrix<float, 6, 6>::Identity();
  name_ = "camera";
  cloud_number_= 0;
}

void
RGBID_SLAM::CameraView::setViewerPose (const Eigen::Affine3d& viewer_pose_d)
{
  boost::mutex::scoped_lock lock(display_mutex_);
  Eigen::Affine3f viewer_pose = viewer_pose_d.cast<float>();
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  cloud_viewer_.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}



void 
RGBID_SLAM::CameraView::updateCamera(const Eigen::Affine3d &new_pose)
{
  bool not_first_pose_flag = removeCamera(); 
  camera_pose_ = new_pose.cast<float>(); 
  
  {
    drawCamera(camera_pose_, 1.0, 0.0, 0.0);  
  }
      
  //drawMotionUncertainty(cov_mat_, camera_pose_, 1.0, 1.0, 1.0) ;   
}

  
void
RGBID_SLAM::CameraView::updateKeyframe(const Eigen::Affine3d &last_KF_pose, int kf_id)
{
  //removeKeyframe(kf_id);
  drawKeyframe(last_KF_pose.cast<float>(), kf_id, 0.0, 0.0, 1.0, 0.25);
}
  

  
void 
RGBID_SLAM::CameraView::updateTrajectory(pcl::PointCloud<pcl::PointXYZ>::Ptr& trajectory)
{
  std::string trajectory_id = "trajectory";
  cloud_viewer_.removeShape(trajectory_id);
  cloud_viewer_.addPolygon<pcl::PointXYZ>(trajectory, 0.0, 1.0, 0.0, trajectory_id);
}

void 
RGBID_SLAM::CameraView::updatePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_point_cloud, int kf_id, Eigen::Affine3d pc_pose)
{
  Eigen::Affine3f pc_pose_f = pc_pose.cast<float>();
  std::string cloud_id = "PointCloud";  
  std::ostringstream convert;
  convert << kf_id;  
  cloud_id.append(convert.str());
  
  if (!cloud_viewer_.updatePointCloudPose(cloud_id, pc_pose_f))
  {
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(new_point_cloud);
    cloud_viewer_.addPointCloud(new_point_cloud, rgb, cloud_id); 
    cloud_viewer_.updatePointCloudPose(cloud_id, pc_pose_f);
  }
}

void
RGBID_SLAM::CameraView::spinOnce(int t)
{
  boost::mutex::scoped_lock lock(display_mutex_);
  cloud_viewer_.spinOnce(t);
}



inline void
RGBID_SLAM::CameraView::drawMotionUncertainty(const Eigen::Matrix<float,6,6> cov_mat, const Eigen::Affine3f& pose, double r, double g, double b)
{
  //boost::mutex::scoped_lock lock(display_mutex_);
  
  Eigen::Matrix3f A = cov_mat.block<3,3>(0,0);
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(A,Eigen::ComputeThinU);
  Eigen::Vector3f singval_A = svd.singularValues();
  Eigen::Matrix3f singvec_A = svd.matrixU();
  Eigen::Matrix3f singval_A_mat = Eigen::Matrix3f::Identity();
  singval_A_mat(0,0) = singval_A(0);
  singval_A_mat(1,1) = singval_A(1);
  singval_A_mat(2,2) = singval_A(2);
  Eigen::Affine3f trafo;
  Eigen::Matrix3f trafomat = 10000.*singvec_A*singval_A_mat.cwiseSqrt();
  trafo.linear() = trafomat;
  //std::cout << trafo.linear() << std::endl;
  //std::cout << trafo.translation() << std::endl;
  pcl::PointXYZ p0x, p1x, p0y, p1y, p0z, p1z;
  p0x.x = -1.; p0x.y = 0.;  p0x.z = 0.;
  p1x.x = 1.;  p1x.y = 0.;  p1x.z = 0.;
  p0y.x = 0.;  p0y.y = -1.; p0y.z = 0.;
  p1y.x = 0.;  p1y.y = 1.;  p1y.z = 0.;
  p0z.x = 0.;  p0z.y = 0.;  p0z.z = -1.;
  p1z.x = 0.;  p1z.y = 0.;  p1z.z = 1.;
  
  p0x = pcl::transformPoint (p0x, trafo);
  p1x = pcl::transformPoint (p1x, trafo);
  p0y = pcl::transformPoint (p0y, trafo);
  p1y = pcl::transformPoint (p1y, trafo);
  p0z = pcl::transformPoint (p0z, trafo);
  p1z = pcl::transformPoint (p1z, trafo);
  
  p0x = pcl::transformPoint (p0x, pose);
  p1x = pcl::transformPoint (p1x, pose);
  p0y = pcl::transformPoint (p0y, pose);
  p1y = pcl::transformPoint (p1y, pose);
  p0z = pcl::transformPoint (p0z, pose);
  p1z = pcl::transformPoint (p1z, pose);
  
  
  
  std::stringstream ss;
  ss.str ("");
  ss << name_ << "_uncline1";
  cloud_viewer_.addLine (p0x, p1x, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_uncline2";
  cloud_viewer_.addLine (p0y, p1y, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_uncline3";
  cloud_viewer_.addLine (p0z, p1z, r, g, b, ss.str ());

}
  
  
void 
RGBID_SLAM::CameraView::drawCamera (const Eigen::Affine3f& pose, double r, double g, double b, double s)
{
  //boost::mutex::scoped_lock lock(display_mutex_);
  double focal = 575;
  double height = 480;
  double width = 640;
  
  // create a 5-point visual for each camera
  pcl::PointXYZ p1, p2, p3, p4, p5;
  p1.x=0; p1.y=0; p1.z=0;
  double angleX = RAD2DEG (2.0 * atan (width / (2.0*focal)));
  double angleY = RAD2DEG (2.0 * atan (height / (2.0*focal)));
  double dist = 0.75;
  double minX, minY, maxX, maxY;
  maxX = dist*tan (atan (width / (2.0*focal)));
  minX = -maxX;
  maxY = dist*tan (atan (height / (2.0*focal)));
  minY = -maxY;
  p2.x=s*minX; p2.y=s*minY; p2.z=s*dist;
  p3.x=s*maxX; p3.y=s*minY; p3.z=s*dist;
  p4.x=s*maxX; p4.y=s*maxY; p4.z=s*dist;
  p5.x=s*minX; p5.y=s*maxY; p5.z=s*dist;
  p1=pcl::transformPoint (p1, pose);
  p2=pcl::transformPoint (p2, pose);
  p3=pcl::transformPoint (p3, pose);
  p4=pcl::transformPoint (p4, pose);
  p5=pcl::transformPoint (p5, pose);
  std::stringstream ss;
  ss.str ("");
  ss << name_ << "_line1";
  cloud_viewer_.addLine (p1, p2, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line2";
  cloud_viewer_.addLine (p1, p3, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line3";
  cloud_viewer_.addLine (p1, p4, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line4";
  cloud_viewer_.addLine (p1, p5, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line5";
  cloud_viewer_.addLine (p2, p5, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line6";
  cloud_viewer_.addLine (p5, p4, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line7";
  cloud_viewer_.addLine (p4, p3, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line8";
  cloud_viewer_.addLine (p3, p2, r, g, b, ss.str ());    
}

void
RGBID_SLAM::CameraView::removeShapes ()
{
  //boost::mutex::scoped_lock lock(display_mutex_);
  cloud_viewer_.removeAllShapes();
}

bool 
RGBID_SLAM::CameraView::removeCamera ()
{
  //boost::mutex::scoped_lock lock(display_mutex_);
  bool cam_removed = false;
  cloud_viewer_.removeShape (name_);
  std::stringstream ss;
  ss.str ("");
  ss << name_ << "_line1";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line2";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line3";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line4";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line5";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line6";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line7";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line8";
  cam_removed = cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_uncline1";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_uncline2";
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_uncline3";
  return cam_removed;
}
  
void 
RGBID_SLAM::CameraView::drawKeyframe (const Eigen::Affine3f& pose, int kf_id, double r, double g, double b, double s)
{
  //boost::mutex::scoped_lock lock(display_mutex_);
  double focal = 575;
  double height = 480;
  double width = 640;
  
  // create a 5-point visual for each camera
  pcl::PointXYZ p1, p2, p3, p4, p5;
  p1.x=0; p1.y=0; p1.z=0;
  double angleX = RAD2DEG (2.0 * atan (width / (2.0*focal)));
  double angleY = RAD2DEG (2.0 * atan (height / (2.0*focal)));
  double dist = 0.75;
  double minX, minY, maxX, maxY;
  maxX = dist*tan (atan (width / (2.0*focal)));
  minX = -maxX;
  maxY = dist*tan (atan (height / (2.0*focal)));
  minY = -maxY;
  p2.x=s*minX; p2.y=s*minY; p2.z=s*dist;
  p3.x=s*maxX; p3.y=s*minY; p3.z=s*dist;
  p4.x=s*maxX; p4.y=s*maxY; p4.z=s*dist;
  p5.x=s*minX; p5.y=s*maxY; p5.z=s*dist;
  p1=pcl::transformPoint (p1, pose);
  p2=pcl::transformPoint (p2, pose);
  p3=pcl::transformPoint (p3, pose);
  p4=pcl::transformPoint (p4, pose);
  p5=pcl::transformPoint (p5, pose);
  std::stringstream ss;
  ss.str ("");
  ss << name_ << "_line1" << kf_id;
  cloud_viewer_.addLine (p1, p2, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line2" << kf_id;
  cloud_viewer_.addLine (p1, p3, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line3" << kf_id;
  cloud_viewer_.addLine (p1, p4, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line4" << kf_id;
  cloud_viewer_.addLine (p1, p5, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line5" << kf_id;
  cloud_viewer_.addLine (p2, p5, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line6" << kf_id;
  cloud_viewer_.addLine (p5, p4, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line7" << kf_id;
  cloud_viewer_.addLine (p4, p3, r, g, b, ss.str ());
  ss.str ("");
  ss << name_ << "_line8" << kf_id;
  cloud_viewer_.addLine (p3, p2, r, g, b, ss.str ()); 
     
}

bool
RGBID_SLAM::CameraView::removeKeyframe (int kf_id)
{
  //boost::mutex::scoped_lock lock(display_mutex_);
  bool kf_removed = false;
  cloud_viewer_.removeShape (name_);
  std::stringstream ss;
  ss.str ("");
  ss << name_ << "_line1" << kf_id;
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line2" << kf_id;
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line3" << kf_id;
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line4" << kf_id;
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line5" << kf_id;
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line6" << kf_id;
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line7" << kf_id;
  cloud_viewer_.removeShape (ss.str ());
  ss.str ("");
  ss << name_ << "_line8" << kf_id;
  
  kf_removed = cloud_viewer_.removeShape (ss.str ());
  
  return kf_removed;
}


RGBID_SLAM::VisualizationManager::VisualizationManager()
{
  cols_ = 640;
  rows_ = 480;

  camera_viewer_.setViewer(0,0,"3D world");
  trajectory_.reset(new pcl::PointCloud<pcl::PointXYZ>) ;
  
  //camera_viewer_.viewer_pose_ = Eigen::Translation3f (t_viewer) * Eigen::AngleAxisf (R_viewer);
  //setViewerPose(camera_viewer_.cloud_viewer_, camera_viewer_.viewer_pose_);
  
  Eigen::Matrix3f R_viewer = Eigen::Matrix3f::Identity ();// * AngleAxisf( pcl::deg2rad(-90.f), Vector3f::UnitX());
  Eigen::Vector3f t_viewer = Eigen::Vector3f (0.f, 0.f, -2.f);
  Eigen::Affine3f viewer_pose =  Eigen::Translation3f (t_viewer) * Eigen::AngleAxisf (R_viewer);
  camera_viewer_.setViewerPose(viewer_pose.cast<double>()); 
  
  scene_viewer_.setViewer(0,700,"Integrated depth view");
  //intensity_warped_viewer_.setViewer(640,700,0.f,255.f,"warped intensity view");
  //segmentation_viewer_.setViewer(cols_,700,"RGB view");
  intensity_viewer_.setViewer(cols_,700,0.f,255.f,"RGB view");
  negentropy_viewer_.setViewer(2*cols_,700,"Negentropy view");
  scene_view_.resize(cols_*rows_);
  intensity_view_.resize(cols_*rows_,0.f);
  negentropy_view_.resize(cols_*rows_);
  segmentation_view_.resize(cols_*rows_);
  depthinv_view_.resize(cols_*rows_,0.f);
  
  
  redraw_pointcloud_= false;
  redraw_trajectory_= false;
  redraw_camera_= false;
  redraw_all_pointclouds_= false;
  redraw_scene_view_= false;
  redraw_keyframe_ = false;
}

RGBID_SLAM::VisualizationManager::~VisualizationManager()
{  
}

        
void 
RGBID_SLAM::VisualizationManager::start()
{
  stop_ = false;
  drawing_thread_.reset(new boost::thread(boost::ref(*this)));
}

void 
RGBID_SLAM::VisualizationManager::stop()
{
  stop_ = true;
}

void 
RGBID_SLAM::VisualizationManager::getPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr whole_point_cloud)
{  
  IdToPoseMap::iterator id2pose_map_it;        
  IdToPointCloudMap::const_iterator id2pointcloud_map_it;
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr aux_point_cloud;  
  aux_point_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  
  for (id2pose_map_it = id2pose_map_.begin(); 
       id2pose_map_it != id2pose_map_.end();
       id2pose_map_it++)
  {
    int kf_id = (*id2pose_map_it).first;
    Eigen::Affine3d pose = (*id2pose_map_it).second;
    
    id2pointcloud_map_it = id2pointcloud_map_.find(kf_id);
    
    if (id2pointcloud_map_it != id2pointcloud_map_.end())
    {
      aux_point_cloud->clear();
      copyPointCloud((*id2pointcloud_map_it).second, aux_point_cloud);  
      (*id2pointcloud_map_it).second->clear();
      alignPointCloud(aux_point_cloud, pose.linear(), pose.translation());   
      *whole_point_cloud += *aux_point_cloud;
    }          
  }  
  
  
}

void 
RGBID_SLAM::VisualizationManager::refresh()
{
  if (visodo_ptr_->camera_pose_has_changed_ == true)
  {
    new_pose_ = visodo_ptr_->getSharedCameraPose();
    redraw_camera_ = true;
  }
  
  if (visodo_ptr_->scene_view_has_changed_ == true)
  {
    boost::mutex::scoped_lock lock(visodo_ptr_->mutex_scene_view_);
    
    scene_view_ = visodo_ptr_->scene_view_;
    intensity_view_ = visodo_ptr_->intensity_view_;
    depthinv_view_ = visodo_ptr_->depthinv_view_;
    
    redraw_scene_view_ = true;
    visodo_ptr_->scene_view_has_changed_ = false;
  }
  
  if (keyframe_manager_ptr_->trajectory_has_changed_ == true)
  {
    boost::mutex::scoped_lock lock(keyframe_manager_ptr_->mutex_odometry_);
    
    copyTrajectory(keyframe_manager_ptr_->poses_);
    
    keyframe_manager_ptr_->trajectory_has_changed_ = false;
    //redraw_trajectory_ = true;
  }
  
  if (keyframe_manager_ptr_->new_keyframe_has_changed_ == true)
  {
    boost::mutex::scoped_lock lock(keyframe_manager_ptr_->mutex_new_keyframe_);
    
    kf_id_ = keyframe_manager_ptr_->new_kf_id_;
    last_KF_pose_ = keyframe_manager_ptr_->new_keyframe_pose_;  
    
    std::pair<IdToPoseMap::iterator,bool> is_new_id;
    is_new_id = id2pose_map_.insert(std::make_pair(kf_id_, last_KF_pose_));  
    
    //segmentation_view_ = keyframe_manager_ptr_->rgb_image_with_keypoints_;
    memcpy ((PixelRGB*)&(segmentation_view_[0]), &(keyframe_manager_ptr_->rgb_image_with_keypoints_.data[0]), 
            keyframe_manager_ptr_->rgb_image_with_keypoints_.cols * keyframe_manager_ptr_->rgb_image_with_keypoints_.rows*sizeof(PixelRGB));
    negentropy_view_ = keyframe_manager_ptr_->negentropy_view_;
    
    keyframe_manager_ptr_->new_keyframe_has_changed_ = false;
    redraw_keyframe_ = true;
  }
  
  if (keyframe_manager_ptr_->new_pointcloud_has_changed_ == true)
  {
    boost::mutex::scoped_lock lock(keyframe_manager_ptr_->mutex_new_pointcloud_);
    
    new_pc_id_ = keyframe_manager_ptr_->new_pointcloud_id_;   
    
    std::pair<IdToPointCloudMap::iterator,bool> is_new_id;
        
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr aux_ptr;
    aux_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>) ;
    
    copyPointCloud(keyframe_manager_ptr_->new_pointcloud_, aux_ptr);
    
    is_new_id = id2pointcloud_map_.insert(std::make_pair(new_pc_id_, aux_ptr));
    
    keyframe_manager_ptr_->new_pointcloud_has_changed_ = false;
    redraw_pointcloud_ = true;
  }
  
  if (keyframe_manager_ptr_->pose_graph_has_changed_ == true)
  {
    IdToPoseMap::iterator id2pose_map_it;
  
    for (int i=0; i<keyframe_manager_ptr_->keyframes_list_.size(); i++)
    {
      id2pose_map_it = id2pose_map_.find(keyframe_manager_ptr_->keyframes_list_[i]->kf_id_);
      if (id2pose_map_it != id2pose_map_.end())
      {
        id2pose_map_[keyframe_manager_ptr_->keyframes_list_[i]->kf_id_] = keyframe_manager_ptr_->keyframes_list_[i]->getPose();
      } 
    }
    
    keyframe_manager_ptr_->pose_graph_has_changed_ = false;
    redraw_all_pointclouds_ = true;
  }
  
  if (redraw_scene_view_)
  {
    scene_viewer_.viewerRGB_.showRGBImage (reinterpret_cast<unsigned char*> (&scene_view_[0]), cols_, rows_);
    intensity_viewer_.viewerFloat_.showFloatImage(&intensity_view_[0], cols_, rows_, intensity_viewer_.min_val_, intensity_viewer_.max_val_,true);
    redraw_scene_view_ = false;
  }
    
  if (redraw_camera_)
  {
    camera_viewer_.updateCamera(new_pose_);
    redraw_camera_ = false;
  }
  
  if (redraw_keyframe_)
  {
    negentropy_viewer_.viewerRGB_.showRGBImage (reinterpret_cast<unsigned char*> (&negentropy_view_[0]), cols_, rows_);
    //segmentation_viewer_.viewerRGB_.showRGBImage (reinterpret_cast<unsigned char*> (&segmentation_view_[0]), cols_, rows_);
    redraw_keyframe_ = false;
  }
  
  if (redraw_pointcloud_)
  {
    IdToPoseMap::iterator id2pose_map_it;
    
    id2pose_map_it = id2pose_map_.find(new_pc_id_);
        
    if (id2pose_map_it != id2pose_map_.end())
    {
      camera_viewer_.updatePointCloud(id2pointcloud_map_[new_pc_id_],(*id2pose_map_it).first, (*id2pose_map_it).second);
    }
    
    redraw_pointcloud_ = false;
  }
  
  if (redraw_trajectory_)
  {
    camera_viewer_.updateTrajectory(trajectory_);
    redraw_trajectory_ = false;
  }
  
  if (redraw_all_pointclouds_)
  {
    IdToPoseMap::iterator id2pose_map_it;        
    IdToPointCloudMap::const_iterator id2pointcloud_map_it;
    
    for (id2pose_map_it = id2pose_map_.begin(); 
         id2pose_map_it != id2pose_map_.end();
         id2pose_map_it++)
    {
      int kf_id = (*id2pose_map_it).first;
      Eigen::Affine3d pose = (*id2pose_map_it).second;
      
      //camera_viewer_.updateKeyframe(pose, kf_id);    
      
      id2pointcloud_map_it = id2pointcloud_map_.find(kf_id);
      
      if (id2pointcloud_map_it != id2pointcloud_map_.end())
      {
        camera_viewer_.updatePointCloud((*id2pointcloud_map_it).second, kf_id, pose);
      }          
    }  
    
    redraw_all_pointclouds_ = false;
  }
  
  camera_viewer_.spinOnce(1);
  //boost::this_thread::sleep (boost::posix_time::millisec (20)); 
}
  
  
  

void 
RGBID_SLAM::VisualizationManager::copyTrajectory(const std::vector<Pose> &cam_poses)
{
  trajectory_->clear();
  
  for (int i=0; i<cam_poses.size(); i++)
  {
    Eigen::Vector3d p_eigen = cam_poses[i].translation_;
    
    pcl::PointXYZ cam_position;
    cam_position.x = p_eigen[0];
    cam_position.y = p_eigen[1];
    cam_position.z = p_eigen[2];
    
    trajectory_->push_back(cam_position); 
  }   
}
  
void 
RGBID_SLAM::VisualizationManager::operator() ()
{
  while (!stop_)
    refresh();  
}
  
 
  
  
   
