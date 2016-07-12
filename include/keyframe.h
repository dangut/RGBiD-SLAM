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

#ifndef KEYFRAME_HPP_
#define KEYFRAME_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <boost/thread/thread.hpp>

//#include "internal.h"
#include "cloud_segmenter.h"
#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"

//#include "float3_operations.h"
//#include "../ThirdParty/pcl_gpu_containers/include/device_array.h"
#include "types.h"
#include <opencv2/opencv.hpp>

  
namespace RGBID_SLAM 
{  
  struct Keyframe
  {           
    Keyframe(Eigen::Matrix3d K, double kd1, double kd2, double kd3, double kd4, double kd5,
              Eigen::Matrix3d rotation, Eigen::Vector3d translation, Eigen::Matrix3d rotation_rel, Eigen::Vector3d translation_rel, int id, int cols = 640, int rows = 480, int bow_histogram_list_size= 1);
    ~Keyframe();
    
    std::vector<PixelRGB> colors_;
    std::vector<float> depthinv_;
    std::vector<float> normals_;
    std::vector<unsigned char> overlap_mask_;
    
    Eigen::Matrix3d K_;
    float kd1_,kd2_, kd3_, kd4_, kd5_;     
    int id_;
    int cols_;
    int rows_;
    int kf_id_;
    
    Eigen::Affine3d pose_rel_me2next_;
    Eigen::Matrix3d Kinv_;
    
    float octave_scale_;
    float sigma_pixel_;
    float sigma_depthinv_;    
    
    cv::Mat rgb_image_;
    cv::Mat grey_image_;
    cv::Mat depthinv_image_;
    cv::Mat normals_image_;
    cv::Mat overlap_mask_image_;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_novel_;
    
    std::vector<Superjut> superjuts_list_;    
    
    std::vector<DBoW2::BowVector> bow_histogram_list_;   
    std::vector<DBoW2::FeatureVector> bow_feature_vector_list_;
    int bow_histogram_list_size_;
    
    std::vector<cv::KeyPoint> keypoints_; 
    std::vector<cv::Mat> descriptors_; //or simply cv::Mat??
    std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > points3D_;
    std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > points3D_cov_;
    std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > points2D_;
    std::vector<Eigen::Matrix2d,Eigen::aligned_allocator<Eigen::Matrix2d> > points2D_cov_;
    
    void setPose(const Eigen::Affine3d& new_pose);
    void setPose(const Eigen::Matrix3d& new_rot, const Eigen::Vector3d& new_trans);
    
    Eigen::Affine3d getPose();
    
    void wrapDataInImages();
    
    void freeUnneedMemory();
    void lift2DKeypointsto3DPoints();      
    void lift2DKeypointsto3DPointsWithCovariance();
    
    void convert2DKeypointsWithCovariance();
    
    bool unprojectPoint(Eigen::Vector3d& X, double d, const Eigen::Vector2d& p);
    bool projectPoint(Eigen::Vector2d& p, const Eigen::Vector3d& X);
    void unprojectionJacobianAtPoint(const Eigen::Vector2d& p, double d, Eigen::Matrix<double,3,3> &unprojJ );
    void projectionJacobianAtPoint(const Eigen::Vector3d& X, Eigen::Matrix<double,2,3> &projJ );
    
    bool undistortPoint(Eigen::Vector2d& p_und, const Eigen::Vector2d& p_dist);
    
    inline void getSizeInBytes()
    {
      int size_RGB = colors_.capacity()*sizeof(PixelRGB);
      int size_depthinv = depthinv_.capacity()*sizeof(float);
      int size_normals = normals_.capacity()*sizeof(float);
      int size_juts = superjuts_list_.capacity()*sizeof(Superjut);
      int size_keypoints = keypoints_.capacity()*sizeof(cv::KeyPoint);
      int size_descriptors = descriptors_.capacity()*32*sizeof(char);
      int size_Bow_hist = bow_histogram_list_[0].size()*sizeof(DBoW2::BowVector);
      int size_Bow_feat = bow_feature_vector_list_[0].size()*sizeof(DBoW2::FeatureVector);
      int size_points3D = points3D_.capacity()*sizeof(Eigen::Vector3d);
      
      int total_size =  size_RGB +
                        size_depthinv +
                        size_normals +
                        size_juts +
                        size_keypoints +
                        size_descriptors +
                        size_Bow_hist +
                        size_Bow_feat + 
                        size_points3D;
                        
      std::cout << "Size XYZRGB: " << sizeof(pcl::PointXYZRGB) << std::endl;
      std::cout << "Total size(MB): " << total_size / 1024.f / 1024.f << std::endl;
      std::cout << "  RGB size(MB): " << size_RGB / 1024.f / 1024.f << std::endl;
      std::cout << "  depthinv size(MB): " << size_depthinv / 1024.f / 1024.f << std::endl;
      std::cout << "  normals size(MB): " << size_normals / 1024.f / 1024.f << std::endl;
      std::cout << "  juts size(MB): " << size_juts / 1024.f / 1024.f << std::endl;
      std::cout << "  keypoints size(MB): " << size_keypoints / 1024.f / 1024.f << std::endl;
      std::cout << "  descriptors size(MB): " << size_descriptors / 1024.f / 1024.f << std::endl;
      std::cout << "  BoWhist size(MB): " << size_Bow_hist / 1024.f / 1024.f << std::endl;
      std::cout << "  BoWfeat size(MB): " << size_Bow_feat / 1024.f / 1024.f << std::endl;
      std::cout << "  Points3D size(MB): " << size_points3D / 1024.f / 1024.f << std::endl;
    };

    private:        
      
      Eigen::Affine3d pose_;
      boost::mutex mutex_pose_;
       
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };
  
}

#endif /* VISODO_KEYFRAME_HPP_RGBD_ */
