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

#ifndef FEATUREEXTRACTOR_HPP_
#define FEATUREEXTRACTOR_HPP_

//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>
//#include <pcl/io/ply_io.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <set>

//#include "internal.h"

//#include "float3_operations.h"
#include "types.h"
#include "keyframe.h"
#include "settings.h"


namespace RGBID_SLAM
{    
  class FeatureExtractor
  {
    public:
    
      FeatureExtractor(int nlevels = 8, float scale = 1.2f/*TODO:arg list*/);   
      
      void loadSettings(const Settings& settings);         
      
      void computeKeypointsAndDescriptors(const cv::Mat& img, const cv::Mat& depthinv_map, std::vector<cv::KeyPoint> &keypoints, std::vector<cv::Mat> &descriptors);      
      
      float getScale()  {return scale_;} 
      
    private:
    
      void computeImagePyramid(const cv::Mat& img);        
      
      void extractKeypointsAtLevel(std::vector<cv::KeyPoint> &keypoints, const cv::Mat& depthinv_map, int level);
      
      void computeKeypointsOrientation(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints);
      
      void computeDescriptors(cv::Mat &img, std::vector<cv::Mat> &descriptors, std::vector<cv::KeyPoint> &keypoints);
            
      void correctKeypointsCoordinates(std::vector<cv::KeyPoint> &keypoints, int level);
      
      bool checkDepthConsistency(const cv::Mat &depthinv_map, int x, int y);
    
      std::vector<cv::Mat> img_pyr_;

      int size_ROI_lvl0_;
      int nlevels_;
      float scale_;
      
      float factor_kp_;
      int num_total_keypoints_d_;
      float r_;
      int num_lvl0_keypoints_d_;
      
      int FAST_radius_;
      int desc_radius_;
      
      std::vector<int> umax_;
  };
  
}

#endif /* VISODO_FEATUREEXTRACTOR_HPP_RGBD_ */
