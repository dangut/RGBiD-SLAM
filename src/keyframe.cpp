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
#include "keyframe_manager.h"
#include "util_funcs.h"


RGBID_SLAM::Keyframe::Keyframe (Eigen::Matrix3d K, double kd1, double kd2, double kd3, double kd4, double kd5,
                                    Eigen::Matrix3d rotation, Eigen::Vector3d translation,  Eigen::Matrix3d rotation_rel, Eigen::Vector3d translation_rel, int id,
                                     int cols, int rows, int bow_histogram_list_size): K_(K), id_(id), cols_(cols), rows_(rows), bow_histogram_list_size_(bow_histogram_list_size)
{
  superjuts_list_.clear();
  
  point_cloud_novel_.reset(new pcl::PointCloud<pcl::PointXYZRGB>) ;
  
  kd1_ = kd1;
  kd2_ = kd2;
  kd3_ = kd3;
  kd4_ = kd4;
  kd5_ = kd5;
  
  //kd1_ = 0.03192f;
  //kd2_ = -0.08934f;
  //kd3_ = 0.f;
  
  ////////////////fr2
  //K_(0,0) = 520.9f;
  //K_(1,1) = 521.0f;
  //K_(0,2) = 325.1f;
  //K_(1,2) = 249.7f;
  
  //kd1_ = 0.2312f;
  //kd2_ = -0.7849f;
  //kd3_ = 0.9172;
  ///////////////
  
  ////////////////fr1
  //K_(0,0) = 517.3f;
  //K_(1,1) = 516.5f;
  //K_(0,2) = 318.6f;
  //K_(1,2) = 255.3f;
  
  //kd1_ = 0.2624f;
  //kd2_ = -0.9531f;  
  //kd3_ = 1.1633f;
  ///////////////
  
  pose_.linear()  = rotation;
  pose_.translation() = translation;
  pose_rel_me2next_.linear()  = rotation_rel;
  pose_rel_me2next_.translation() = translation_rel;
  Kinv_ = K_.inverse();
}

RGBID_SLAM::Keyframe::~Keyframe ()
{ 
  superjuts_list_.clear();
}

void RGBID_SLAM::Keyframe::setPose(const Eigen::Affine3d& new_pose)
{
  boost::mutex::scoped_lock lock(mutex_pose_);  
  pose_ = new_pose;
  return;
}

void RGBID_SLAM::Keyframe::setPose(const Eigen::Matrix3d& new_rot, const Eigen::Vector3d& new_trans)
{
  boost::mutex::scoped_lock lock(mutex_pose_);
  pose_.linear() = new_rot;
  pose_.translation() = new_trans;
  return;
}

Eigen::Affine3d RGBID_SLAM::Keyframe::getPose()
{
  boost::mutex::scoped_lock lock(mutex_pose_);
  return pose_;
}

void RGBID_SLAM::Keyframe::wrapDataInImages()
{
  //Wrap depth, rgb and normal data in opencv matrices
  
  //pcl::ScopeTime t3 ("wrap in opencv");
  rgb_image_ = cv::Mat(rows_, cols_, CV_8UC3, &colors_[0], CV_ELEM_SIZE(CV_8UC3)*cols_);
  depthinv_image_ = cv::Mat(rows_, cols_, CV_32F, &depthinv_[0], CV_ELEM_SIZE(CV_32F)*cols_);
  normals_image_ = cv::Mat(3*rows_, cols_, CV_32F, &normals_[0], CV_ELEM_SIZE(CV_32F)*cols_); 
  overlap_mask_image_ = cv::Mat(rows_, cols_, CV_8UC1, &overlap_mask_[0], CV_ELEM_SIZE(CV_8UC1)*cols_);
}

void RGBID_SLAM::Keyframe::freeUnneedMemory()
{
  normals_image_.release();
  normals_.clear();
  normals_ = std::vector<float>();
  
  //rgb_image_.release();
  //colors_.clear();
  //colors_ = std::vector<PixelRGB>();
  //depthinv_image_.release();
  //depthinv_.clear();
  //depthinv_ = std::vector<float>();
  
  grey_image_.release();
}

    
void RGBID_SLAM::Keyframe::lift2DKeypointsto3DPoints()
{
  points3D_.clear();
  points3D_.reserve(keypoints_.size());
  
  for (std::vector<cv::KeyPoint>::iterator kp_it = keypoints_.begin(); kp_it != keypoints_.end(); )
  {
    Eigen::Vector2d p_dist(kp_it->pt.x, kp_it->pt.y);
    Eigen::Vector2d p_im;
    if (undistortPoint(p_im, p_dist))
    {
      int y = (int) (p_im[1]+0.5);
      int x = (int) (p_im[0]+0.5);
      double d = 1.f / depthinv_image_.ptr<float>(y)[x];
      Eigen::Vector3d X_3D;
      unprojectPoint(X_3D,d,p_im);
      points3D_.push_back(X_3D);
      kp_it++;
    }
    else
    {
      kp_it = keypoints_.erase(kp_it);
    }
  }
}


void RGBID_SLAM::Keyframe::lift2DKeypointsto3DPointsWithCovariance()
{
  points3D_.clear();
  points3D_.reserve(keypoints_.size());
  points3D_cov_.clear();
  points3D_cov_.reserve(keypoints_.size());
  
  for (std::vector<cv::KeyPoint>::iterator kp_it = keypoints_.begin(); kp_it != keypoints_.end(); )
  {
    Eigen::Vector2d p_dist(kp_it->pt.x, kp_it->pt.y);
    Eigen::Vector2d p_im;
    if (undistortPoint(p_im, p_dist))
    {
      int y = (int) (p_im[1]+0.5);
      int x = (int) (p_im[0]+0.5);
      double d = 1.f / depthinv_image_.ptr<float>(y)[x];
      Eigen::Vector3d X_3D;
      unprojectPoint(X_3D,d,p_im);
      points3D_.push_back(X_3D);
      
      Eigen::Matrix3d unprojJ;
      unprojectionJacobianAtPoint(p_im, d, unprojJ);
      
      //Eigen::Matrix2d cov_u;    
      //cov_u << sigma2_u, 0.f, 0.f, sigma2_u;
      float kp_scale = std::pow(octave_scale_, kp_it->octave);
      float sigma2_u_level = kp_scale*kp_scale*sigma_pixel_*sigma_pixel_;
      float sigma2_depthinv = sigma_depthinv_*sigma_depthinv_;
      
      Eigen::Matrix3d cov_u_and_invdepth;// = Eigen::Matrix3d::Identity();
      cov_u_and_invdepth << sigma2_u_level, 0.f, 0.f, 0.f, sigma2_u_level, 0.f, 0.f, 0.f, sigma2_depthinv;
    
      Eigen::Matrix<double,3,3> cov_X3D =  unprojJ*cov_u_and_invdepth*unprojJ.transpose();    
      
      points3D_cov_.push_back(cov_X3D);
      kp_it++;
    }
    else
    {
      kp_it = keypoints_.erase(kp_it);
    }
  }
}
    
void RGBID_SLAM::Keyframe::convert2DKeypointsWithCovariance() 
{
  points2D_.clear();
  points2D_.reserve(keypoints_.size());
  points2D_cov_.clear();
  points2D_cov_.reserve(keypoints_.size());
  
  for (std::vector<cv::KeyPoint>::const_iterator kp_it = keypoints_.begin(); kp_it != keypoints_.end(); kp_it++)
  {
    Eigen::Vector2d p_im(kp_it->pt.x, kp_it->pt.y);
    points2D_.push_back(p_im);
    
    float kp_scale = std::pow(octave_scale_, kp_it->octave);
    float sigma2_u_level = kp_scale*kp_scale*sigma_pixel_*sigma_pixel_;
    
    Eigen::Matrix2d cov_u;// = Eigen::Matrix3d::Identity();
    cov_u << sigma2_u_level, 0.f, 0.f, sigma2_u_level;
    
    points2D_cov_.push_back(cov_u);
  }
  
}
bool RGBID_SLAM::Keyframe::undistortPoint(Eigen::Vector2d& p_und, const Eigen::Vector2d& p_dist)
{
  Eigen::Vector3d p_inP2(p_dist[0],p_dist[1],1.f);
  
  Eigen::Vector3d m_dist = Kinv_*p_inP2;
  
  Eigen::Vector3d m_undist = m_dist;
  
  for (int i=0; i<20; i++)
  {
    double r2 = m_undist[0]*m_undist[0]+m_undist[1]*m_undist[1];
    double r4 = r2*r2;
    double r6 = r4*r2;
    double radial_factor = (1.f + kd1_*r2+kd2_*r4+kd5_*r6) ;
    Eigen::Vector3d tang_dist;
    tang_dist[0] = 2.f*kd3_*m_undist[0]*m_undist[1] +  kd4_*(r2 + 2.f*m_undist[0]*m_undist[0]);
    tang_dist[1] = 2.f*kd4_*m_undist[0]*m_undist[1] +  kd3_*(r2 + 2.f*m_undist[1]*m_undist[1]);
    
     m_undist.head(2) = (m_dist.head(2) - tang_dist) / radial_factor;
  }
  
  p_inP2 = K_*m_undist;
  p_und[0] = p_inP2[0];
  p_und[1] = p_inP2[1];
  
  return (p_und[0] > 0.f) && (p_und[0] < (float(cols_ - 1))) && (p_und[1] > 0.f) && (p_und[1] < (float(rows_ - 1)));
  
}

bool RGBID_SLAM::Keyframe::unprojectPoint(Eigen::Vector3d& X, double d, const Eigen::Vector2d& p)
{
  Eigen::Vector3d p_inP2(p[0],p[1],1.f);
  
  Eigen::Vector3d m_dist = Kinv_*p_inP2;
  
  Eigen::Vector3d m_undist = m_dist;
  
  //for (int i=0; i<20; i++)
  //{
    //double r2 = m_undist[0]*m_undist[0]+m_undist[1]*m_undist[1];
    //double r4 = r2*r2;
    //double r6 = r4*r2;
    //double factor = (1.f / (1.f + kd1_*r2+kd2_*r4+kd3_*r6) );
    //m_undist = factor*m_dist;
  //}
  
  X = d*m_undist;
  
  return true;
}

void RGBID_SLAM::Keyframe::unprojectionJacobianAtPoint(const Eigen::Vector2d& p, double d, Eigen::Matrix<double,3,3> &unprojJ )
{
  Eigen::Vector3d p_inP2;
  p_inP2 << p[0], p[1], 1.f;
  Eigen::Vector3d X;
  X = d*Kinv_*p_inP2;
  
  double inv_d = (1.f/d);
  
  Eigen::Matrix<double,3,2> dX_by_dp = inv_d*Kinv_.block<3,2>(0,0);
  Eigen::Matrix<double,3,1> dX_by_dinvd = -inv_d*inv_d*Kinv_*p_inP2;
  
  unprojJ.block<3,2>(0,0) = dX_by_dp;
  unprojJ.block<3,1>(0,2) = dX_by_dinvd;  
}

bool RGBID_SLAM::Keyframe::projectPoint(Eigen::Vector2d& p, const Eigen::Vector3d& X)
{
  Eigen::Vector3d m_undist = (1.f/X[2])*X;
  
  double r2 = m_undist[0]*m_undist[0]+m_undist[1]*m_undist[1];
  double r4 = r2*r2;
  double r6 = r4*r2;
  
  Eigen::Vector3d m_dist = (1.f + kd1_*r2 + kd2_*r4 + kd3_*r6)*m_undist;
  //Eigen::Vector3d m_dist = m_undist;
  
  Eigen::Vector3d p_inP2 = K_*m_dist;
  
  p = p_inP2.head<2>();
  
  return ( (p[0] > 0) && (p[1] > 0) && (p[0] < cols_-1) && (p[1] < rows_-1) );
}

void RGBID_SLAM::Keyframe::projectionJacobianAtPoint(const Eigen::Vector3d& X, Eigen::Matrix<double,2,3> &projJ )
{
  double inv_d = (1.f/X[2]);
  Eigen::Vector3d p_inP2 = inv_d*K_*X;
  Eigen::Vector3d e3;
  e3 << 0.f, 0.f, 1.f;
  
  Eigen::Matrix<double,3,3> dpinP2_by_dX = inv_d*(K_-p_inP2*e3.transpose());
  
  projJ = dpinP2_by_dX.block<2,3>(0,0);
}
