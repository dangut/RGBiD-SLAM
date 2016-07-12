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


#ifndef KEYFRAMEALIGN_HPP_
#define KEYFRAMEALIGN_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <deque>
#include <set>
//#include <boost/thread/thread.hpp>

#include "internal.h"

#include "float3_operations.h"
#include "../ThirdParty/pcl_gpu_containers/include/device_array.h"
#include "types.h"
#include "keyframe.h"
#include "settings.h"

namespace RGBID_SLAM
{        	
  class KeyframeAlign
  {
    public:

      KeyframeAlign();
      ~KeyframeAlign();
      
      void loadSettings(Settings const &settings);
      
      enum { LEVELS = 4 };         
      
      bool alignKeyframes(KeyframePtr& kf_ini, KeyframePtr& kf_end, Eigen::Affine3d& pose_ini2end, Eigen::Matrix<double,6,6>& covariance_ini2end);
      
      bool alignKeyframes(KeyframePtr& kf_ini, KeyframePtr& kf_end, Eigen::Matrix3d& rotation_ini2end, Eigen::Vector3d& translation_ini2end, Eigen::Matrix<double,6,6>& covariance_ini2end);
      
    private:
    
      int finest_level_;
      int numSMs_;      
      
      /** \brief Temporary buffer for ICP */
      DeviceArray2D<RGBID_SLAM::device::float_type> gbuf_;

      /** \brief Buffer to store MLS matrix. */
      DeviceArray<RGBID_SLAM::device::float_type> sumbuf_;    
      
      std::vector<DepthMapf> warped_depthinvs_end_;      
      std::vector<DepthMapf> depthinvs_ini_; 
      std::vector<DepthMapf> depthinvs_end_;
      
      std::vector<DepthMapf> warped_intensities_end_;     
      std::vector<DepthMapf> intensities_ini_; 
      std::vector<DepthMapf> intensities_end_;
      
      std::vector<DepthMapf>  xGradsDepthinv_ini_;
      std::vector<DepthMapf>  yGradsDepthinv_ini_;
      
      std::vector<DepthMapf>  xGradsIntensity_ini_;
      std::vector<DepthMapf>  yGradsIntensity_ini_;
      
      std::vector< DeviceArray<float> > res_depthinvs_;
      std::vector< DeviceArray<float> > res_intensities_;
      std::vector< DeviceArray<float> > res_p2s_distinv_;
      
      DepthMapf  depthinv_warped_in_end_;
      DepthMapf  warped_weight_end_;
      std::vector< DeviceArray2D<float> >  projected_transformed_points_;
      
      std::vector< MapArr > normals_ini_;
      std::vector< MapArr > vertices_ini_;
      std::vector< MapArr > vertices_end_;
      std::vector< MapArr > warped_vertices_end_;
      
      std::vector< DeviceArray2D<float> > inv_dot_vertices_ini_normals_ini_;  
      std::vector< DeviceArray2D<float> > inv_dot_vertices_end_normals_ini_;  
      std::vector< MapArr > normals_ini_in_end_;  
      
      int alignment_iterations_[LEVELS];
      
      Eigen::Affine3d aligned_pose_;
      
      public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  
}

#endif /* KEYFRAMEALIGN_HPP_ */
