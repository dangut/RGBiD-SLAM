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

#ifndef KEYFRAMEMANAGER_HPP_
#define KEYFRAMEMANAGER_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <deque>
#include <set>
#include <boost/thread/thread.hpp>

//#include "internal.h"
#include "cloud_segmenter.h"
#include "loop_closer.h"
#include "feature_extractor.h"

//#include "float3_operations.h"
//#include "../ThirdParty/pcl_gpu_containers/include/device_array.h"
#include "types.h"
#include <opencv2/opencv.hpp>
#include "keyframe.h"
#include "settings.h"

namespace RGBID_SLAM
{        	
  class KeyframeManager
  {
    public:

      KeyframeManager(int Nbins = 80);
      ~KeyframeManager();
      
      void loadSettings(Settings const &settings);
      
      enum { LEVELS = 4 }; 
      
      inline void start()
      { 
        stop_flag_ = false;
        consumer_thread_.reset(new boost::thread(boost::ref(*this)));
        return;
      }
      
      inline void stop()
      {
        stop_flag_ = true;
      }
      
      inline void join()
      {
        consumer_thread_->join();
      }
      
      boost::shared_ptr<boost::thread> consumer_thread_;
      BufferWrapper<KeyframePtr> buffer_keyframes_;
      CloudSegmenterPtr cloud_segmenter_ptr_;
      LoopCloserPtr loop_closer_ptr_;
      FeatureExtractorPtr feature_extractor_ptr_;
      PoseGraphPtr pose_graph_ptr_;      
            
      std::vector<KeyframePtr> keyframes_list_;
      
      KeyframePtr keyframe_new_;
      KeyframePtr keyframe_last_;
      
      boost::mutex mutex_odometry_;
      std::vector<Pose> poses_;
      std::vector<PoseConstraint> constraints_;
      bool trajectory_has_changed_;
      
      boost::mutex mutex_new_pose_graph_;
      bool pose_graph_has_changed_;
      
      boost::mutex mutex_new_keyframe_;
      Eigen::Affine3d new_keyframe_pose_;
      int new_kf_id_;
      bool new_keyframe_has_changed_;
      
      boost::mutex mutex_new_pointcloud_;
      int new_pointcloud_id_;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_pointcloud_;
      bool new_pointcloud_has_changed_;
      
      std::vector<LoopCluster> loop_clusters_;
      
      bool stop_flag_;
      bool loop_found_;
      
      int Nbins_;
      
      void operator() ();  
      
      void performOptimisation();        
      
      //Moved from keyframe
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_;
      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_;
      cv::Mat negentropy_image_;
      cv::Mat segmentation_image_;
      std::vector<int> cloud2image_indices_;
      std::vector<int> image2cloud_indices_;
      std::vector<Superjut> superjuts_list_;    
      ////////////////////////////////////////////
      
      //std::vector<PixelRGB> keypoints_view_;
      std::vector<PixelRGB> negentropy_view_;
      cv::Mat rgb_image_with_keypoints_;
      cv::Mat rgb_image_masked_;
      
      //Log vars
      std::vector<float> segmentation_times_;
      std::vector<float> description_times_;
      std::vector<float> loop_detection_times_;
      std::vector<float> pose_graph_times_;
      std::vector<float> total_times_;
      
    private:
    
      void updatePoses(std::vector<Pose>& poses, std::vector<PoseConstraint>& seq_constraints_new);      
      void processNewKeyframe();      
      void computeLoopClosureEdges();      
      void computeAlignedPointCloud(KeyframePtr& kf, Eigen::Affine3d aligned_pose = Eigen::Affine3d::Identity(), bool align_normals = false /*,Eigen::Affine3d pose_prev2curr*/);
      
      void getRGBNegentropy (PixelRGB color_scale, std::vector<PixelRGB>& negentropy_RGB);
      
      int kf_count_;
      
      float sigma_pixel_;
      float sigma_invdepth_;
      
      Eigen::Affine3d aligned_pose_;
      
      int counter_nofound_;
      int max_counter_nofound_;
      int num_kf_since_last_optim_;
      int max_cluster_unupdated_time_;
      
      public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  
}

#endif /* KEYFRAMEMANAGER_HPP_ */
