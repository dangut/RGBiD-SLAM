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

//#include "internal.h"
//using namespace RGBID_SLAM::device;
//using RGBID_SLAM::device::device_cast;

RGBID_SLAM::KeyframeManager::KeyframeManager(int Nbins):Nbins_(Nbins)
{
  stop_flag_ = false;
  std::cout << "kcloser" << std::endl;
  loop_closer_ptr_.reset(new LoopCloser); //TODO: arg list
  std::cout << "kcloser" << std::endl;
  feature_extractor_ptr_.reset(new FeatureExtractor); //TODO arg list
  std::cout << "extractor" << std::endl;
  pose_graph_ptr_.reset(new PoseGraph);
  std::cout << "kpose graph" << std::endl;
  cloud_segmenter_ptr_.reset(new CloudSegmenter);
  
  std::cout << "kf manager" << std::endl;
  
  buffer_keyframes_.initialise(100);
  
  kf_count_ = 0;
  int cols = 640;
  int rows = 480;
  
  sigma_pixel_ = 0.5;
  sigma_invdepth_ = 0.00025f;
  
  point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>) ;
  new_pointcloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>) ;
  normal_cloud_.reset(new pcl::PointCloud<pcl::Normal>) ;
  
  superjuts_list_.clear();
  
  aligned_pose_.linear() = Eigen::Matrix3d::Identity();
  aligned_pose_.translation() = Eigen::Vector3d(0,0,0);
  
  constraints_.reserve(1000);
  poses_.reserve(1000);
  
  loop_found_ = false;
  counter_nofound_ = 0;
  max_counter_nofound_ = -1;
  max_cluster_unupdated_time_ = 5;
  
  trajectory_has_changed_= false;
  pose_graph_has_changed_= false;
  new_keyframe_has_changed_= false;
}

RGBID_SLAM::KeyframeManager::~KeyframeManager()
{
  std::cout << "destructing loop closer" << std::endl;
  loop_closer_ptr_.reset(); //TODO: arg list
  std::cout << "destructing feature extractor" << std::endl;
  feature_extractor_ptr_.reset(); //TODO arg list
  std::cout << "destructing pose graph" << std::endl;
  pose_graph_ptr_.reset();  
  std::cout << "destructing cloud segmenter" << std::endl;  
  cloud_segmenter_ptr_.reset();
  keyframes_list_.clear();
  buffer_keyframes_.initialise(0);
  constraints_.clear();
  poses_.clear();
  rgb_image_with_keypoints_.release();
  
  point_cloud_->clear();
  normal_cloud_->clear();
  cloud2image_indices_.clear();
  image2cloud_indices_.clear();
  superjuts_list_.clear();
  
}

void
RGBID_SLAM::KeyframeManager::loadSettings(Settings const &settings)
{
  Section keyframe_manager_section;
  
  if (settings.getSection("KEYFRAME_MANAGER",keyframe_manager_section))
  {
    std::cout << "KEYFRAME_MANAGER" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    Entry entry;
    
    if (keyframe_manager_section.getEntry("NORMAL_HISTOGRAM_BINS",entry))
    {
      std::string Nbins_str = entry.getValue();
      
      Nbins_ = atoi(Nbins_str.c_str());
      
      std::cout << "  NORMAL_HISTOGRAM_BINS: " << Nbins_ << std::endl;    
    }
    
    if (keyframe_manager_section.getEntry("STD_PIXEL",entry))
    {
      std::string sigma_pixel_str = entry.getValue();
      
      sigma_pixel_ = atof(sigma_pixel_str.c_str());
      
      std::cout << "  STD_PIXEL: " << sigma_pixel_ << std::endl;    
    }
    
    if (keyframe_manager_section.getEntry("STD_INVDEPTH",entry))
    {
      std::string sigma_invdepth_str = entry.getValue();
      
      sigma_invdepth_ = atof(sigma_invdepth_str.c_str());
      
      std::cout << "  STD_INVDEPTH: " << sigma_invdepth_ << std::endl;    
    }
    
    std::cout << std::endl;
    std::cout << std::endl;
  }  
      
  loop_closer_ptr_->loadSettings(settings);
  feature_extractor_ptr_->loadSettings(settings);
  pose_graph_ptr_->loadSettings(settings);
  cloud_segmenter_ptr_->loadSettings(settings);
}


void
RGBID_SLAM::KeyframeManager::operator() ()
{  
  char loop_file[128];
  std::ofstream output_loop;
  
  buffer_keyframes_.initialise(100);
  
  num_kf_since_last_optim_ = 6;
  
  loop_clusters_.clear();
  
  while (1) 
  {
    //std::cout << "Ini operator0" << std::endl;
    if (buffer_keyframes_.try_pop(keyframe_new_))
    {
      pcl::ScopeTime t1 ("processing new_keyframe");
      
      //Segmentation in juts, extraction of descriptors
      processNewKeyframe();   
      
      std::cout << "KF id: " << keyframe_new_->kf_id_ << std::endl;    
      
      //Loop closure  
      //loop_closer_ptr_->computeBowHistograms(keyframe_new_);  
      {
        pcl::ScopeTime t2("loopDetection");
        for (int i=0; i<loop_clusters_.size(); i++)
        {
          loop_clusters_[i].unupdated_time_++;
        }
        if (loop_closer_ptr_->detectLoopClosures(keyframes_list_, keyframe_new_,loop_clusters_))
          std::cout << "Found a loop!!" << std::endl;
          
        keyframes_list_.push_back(keyframe_new_); 
        keyframe_last_ = keyframe_new_;  
        kf_count_++;
        
        for (std::vector<LoopCluster>::iterator clu_it = loop_clusters_.begin(); clu_it != loop_clusters_.end();)
        {
          if (clu_it->unupdated_time_ >= max_cluster_unupdated_time_)
          {
            clu_it->sortLoopsByInformation();
            
            {
              boost::mutex::scoped_lock lock(mutex_odometry_);
              //constraints_.push_back(clu_it->loop_constraints_[0]); 
              constraints_.insert(constraints_.end(),clu_it->loop_constraints_.begin(),clu_it->loop_constraints_.end());  
            }
            clu_it = loop_clusters_.erase(clu_it);
            loop_found_ = true;
          }
          else
          {
            clu_it++;
          }          
        } 
        loop_detection_times_.push_back(t2.getTime());
      }
      
      num_kf_since_last_optim_++;
      
      {
        pcl::ScopeTime t3 ("pose graph optim");
        if (loop_found_ &&(num_kf_since_last_optim_ > 10))
        {   
          
          performOptimisation(); 
          loop_found_ = false;
          num_kf_since_last_optim_ = 0;   
          
        }  
        pose_graph_times_.push_back(t3.getTime());
      }
      
      total_times_.push_back(t1.getTime());      
    }
    else
    {
      if (stop_flag_)
      {
        pcl::ScopeTime t3 ("pose graph optim");
        //load final constraints in graph and perform final optimisation
        for (std::vector<LoopCluster>::iterator clu_it = loop_clusters_.begin(); clu_it != loop_clusters_.end();clu_it++)
        {
          boost::mutex::scoped_lock lock(mutex_odometry_);
          //constraints_.push_back(clu_it->loop_constraints_[0]); 
          constraints_.insert(constraints_.end(),clu_it->loop_constraints_.begin(),clu_it->loop_constraints_.end());  
        }
        performOptimisation(); 
        pose_graph_times_[pose_graph_times_.size()-1] += t3.getTime();
        break;
      }        
    }
    //std::cout << "End operator" << std::endl;  
  }  
  
  std::cout << "End of keyframe manager thread" << std::endl;
}


void 
RGBID_SLAM::KeyframeManager::performOptimisation()
{
  if (poses_.empty()) return;

  std::cout << "Building graph..." << std::endl
                                     << std::endl;
  //The loop has ended, now we optimise with all the constraints  
  {    
    boost::mutex::scoped_lock lock(mutex_odometry_);
    pose_graph_ptr_->buildGraph(poses_, constraints_);
  }      
  
  std::cout << "Optimising..." << std::endl
                               << std::endl;
  pose_graph_ptr_->optimiseGraph(); 
  std::cout << "Updating poses..." << std::endl
                               << std::endl;
  {
    boost::mutex::scoped_lock lock(mutex_odometry_);
                              
    pose_graph_ptr_->updatePosesAndKeyframes(poses_, keyframes_list_);
    std::cout << "Updating visualizer..." << std::endl
                               << std::endl;
    //trajectory_has_changed_ = true;
  }   
                               
  {
    boost::mutex::scoped_lock lock(mutex_new_pose_graph_);
    pose_graph_has_changed_ = true;
  }
  
  
  
  //updateVisualizerComplete(keyframes_list_, poses_);   
  std::cout << "Optimised..." << std::endl;   
  
  //for (int i=0; i<keyframes_list_.size(); i++)
  //{
    //keyframes_list_[i]->freeUnneedMemory();
  //} 
}




void 
RGBID_SLAM::KeyframeManager::processNewKeyframe()
{  
  //std::cout << "processing new_keyframe" << std::endl;
  
  //Compute point cloud and images from raw data
  keyframe_new_->wrapDataInImages();
  //std::cout << "getting PC" << std::endl;  
  cv::cvtColor(keyframe_new_->rgb_image_, keyframe_new_->grey_image_, CV_BGR2GRAY);
  
  if (kf_count_ > 0)
    keyframe_new_->setPose((keyframes_list_.back()->getPose()*(keyframes_list_.back()->pose_rel_me2next_)));
    
  
  bool align_normals_flag = true;
  
  keyframe_new_->kf_id_ = kf_count_;
  int cols = keyframe_new_->cols_;
  int rows =  keyframe_new_->rows_;
  
  //std::cout << rows << " " << cols << std::endl;
  computeAlignedPointCloud(keyframe_new_, Eigen::Affine3d::Identity(),align_normals_flag);
    
  //Segmentation  
  cloud_segmenter_ptr_->uploadPointCloud(point_cloud_ ,normal_cloud_);
  {
    pcl::ScopeTime t1 ("segmentation");
    cloud_segmenter_ptr_->computeEdgesPixelNeighbors(cloud2image_indices_, image2cloud_indices_, cols, rows);
     
    cloud_segmenter_ptr_->doSegmentation(superjuts_list_);
      
    for (std::vector<Superjut>::iterator it_jut = superjuts_list_.begin(); 
            it_jut != superjuts_list_.end(); it_jut++)
    {
      it_jut->computeHistogramOfNormalsAndEntropy(normal_cloud_, Nbins_);
      //TODO: compute more properties
    }
    
    segmentation_times_.push_back(t1.getTime());
  }
  
  {   
    negentropy_image_ = cv::Mat::zeros(rows, cols, CV_32FC1);
    
    for (std::vector<Superjut>::iterator it_jut = superjuts_list_.begin(); 
          it_jut != superjuts_list_.end(); it_jut++)
    {     
      for (std::set<int>::const_iterator it_point_id = it_jut->cloud_indices_.begin();
            it_point_id != (it_jut->cloud_indices_.end()); it_point_id++)
      {     
        int point_id = (*it_point_id);
        
        float negentropy = (-(*it_jut).normal_entropy_ + 1.f);// * 255.f;
        
        int image_id = cloud2image_indices_[point_id];
        int x = image_id % cols;
        int y = image_id / cols;
        negentropy_image_.ptr<float>(y)[x] =  negentropy;        
      }
    }
    
    negentropy_image_.copyTo(keyframe_new_->negentropy_image_);
  } 
  ////////////////////////////
  
  
  //Keypoints and descriptors
  
  {
    pcl::ScopeTime t3("description");
    feature_extractor_ptr_->computeKeypointsAndDescriptors(keyframe_new_->grey_image_, keyframe_new_->depthinv_image_, keyframe_new_->keypoints_, keyframe_new_->descriptors_);
    keyframe_new_->octave_scale_ = feature_extractor_ptr_->getScale();
    keyframe_new_->sigma_pixel_ = sigma_pixel_;
    keyframe_new_->sigma_depthinv_ = sigma_invdepth_;
    
    keyframe_new_->lift2DKeypointsto3DPointsWithCovariance();    
    keyframe_new_->convert2DKeypointsWithCovariance();
    
    assert(keyframe_new_->points2D_.size() == keyframe_new_->points3D_.size());
    
    keyframe_new_->computeMaskedDescriptors();
    std::cout << "masking bows " << std::endl;
    loop_closer_ptr_->computeBowHistograms(keyframe_new_); 
    
    description_times_.push_back(t3.getTime()); 
  }
  ////////////////////////////////////////////
  
  //mask_RGB = 255*mask_RGB;
  //mask_RGB.copyTo(rgb_image_masked_);
  //TODO: Generate different masked RGB images for every keyframe. keyframe_new_->negentropy_image_
  //for (int y=0;  y<(keyframe_new_->overlap_mask_image_).rows; y++)
  //{
    //for (int x=0; x<(keyframe_new_->overlap_mask_image_).cols; x++)
    //{
      //unsigned char overlap_bool = (keyframe_new_->overlap_mask_image_).ptr<unsigned char>(y)[x];
      //(keyframe_new_->overlap_mask_image_).ptr<unsigned char>(y)[x] = 1-overlap_bool;      
    //}
  //}
  
  //cv::Mat mask_RGB;
  //cv::cvtColor((keyframe_new_->overlap_mask_image_),mask_RGB,cv::COLOR_GRAY2BGR);
  //rgb_image_masked_ = cv::Scalar(0);
  
  
  
  {
    boost::mutex::scoped_lock lock(mutex_new_keyframe_);
    
    cv::drawKeypoints(keyframe_new_->rgb_image_, keyframe_new_->keypoints_, rgb_image_with_keypoints_); 
    
    PixelRGB color;
    color.r = 200; color.g = 0; color.b = 255;
    
    negentropy_view_.resize(negentropy_image_.cols * negentropy_image_.rows);
    getRGBNegentropy(color, negentropy_view_);  
    
    new_keyframe_pose_.linear() = keyframe_new_->getPose().rotation();
    new_keyframe_pose_.translation() = keyframe_new_->getPose().translation();
    new_kf_id_ = keyframe_new_->kf_id_;
    rgb_image_masked_ = cv::Mat::zeros(rows, cols, CV_8UC3);
    rgb_image_with_keypoints_.copyTo(rgb_image_masked_,keyframe_new_->mask_neg_list_[3]);
    new_keyframe_has_changed_ = true;
  }
  
  
  
  //updateVisualizerKeyframe(keyframe_new_->pose_, keyframe_new_->kf_id_, point_cloud_);
  
  //if ((keyframe_new_->kf_id_ % 4) == 0)
  {
    boost::mutex::scoped_lock lock(mutex_new_pointcloud_);
    copyPointCloud(keyframe_new_->point_cloud_novel_, new_pointcloud_);
    new_pointcloud_id_ = keyframe_new_->kf_id_;
    new_pointcloud_has_changed_ = true;
  }
  
}

void RGBID_SLAM::KeyframeManager::computeAlignedPointCloud(KeyframePtr& kf, Eigen::Affine3d aligned_pose, bool align_normals /*,Eigen::Affine3f pose_prev2curr*/)
{
  //pcl::ScopeTime t3 ("compute point cloud");
  
  int rows = kf->rows_;
  int cols = kf->cols_;  
  
  Eigen::Matrix3d Kinv = (kf->K_).inverse();
  
  (kf->point_cloud_novel_)->clear();
  
  point_cloud_->clear();
  normal_cloud_->clear();
  cloud2image_indices_.clear();
  image2cloud_indices_.clear();
  image2cloud_indices_.resize(cols*rows);
  
  Eigen::Matrix3d R_WC = aligned_pose.linear();
  Eigen::Vector3d t_WC = aligned_pose.translation();
  
  //TODO: copy last keyframe and keypose to keyframe manager thread (KF_pose, KF_id, color and invdepth maps);
  int point_id = 0;
  //This point cloud copy is just for visualization
  for (int y = 0; y < rows; y++)
  {
    for (int x = 0; x < cols; x++)
    {          
      //std::cout << "Get depth" << std::endl;
      float d = 1.f / (kf->depthinv_image_).ptr<float>(y)[x];
      
      image2cloud_indices_[y*cols + x] = -1;
         
      if (!isnan(d))
      {
        Eigen::Vector3d p;
        p[0] = (float)(x);
        p[1] = (float)(y);
        p[2] = 1.f;
        Eigen::Vector3d Xcam = d * Kinv * p;        
        
        {
          Eigen::Vector3d Xworld = R_WC*Xcam + t_WC;
          PixelRGB rgb = (kf->rgb_image_).ptr<PixelRGB>(y)[x];
          pcl::PointXYZRGB point_RGB(rgb.r, rgb.g, rgb.b);
          point_RGB.x = Xworld[0]; 
          point_RGB.y = Xworld[1]; 
          point_RGB.z = Xworld[2]; 
          
          if (align_normals)
          {
            pcl::Normal point_normal;
            Eigen::Vector3d ncam;
            
            ncam[0] = (kf->normals_image_).ptr<float>(y)[x];
            ncam[1] = (kf->normals_image_).ptr<float>(y+rows)[x];
            ncam[2] = (kf->normals_image_).ptr<float>(y+2*rows)[x];
            Eigen::Vector3d nworld = R_WC*ncam;
            point_normal.normal_x = nworld[0];
            point_normal.normal_y = nworld[1];
            point_normal.normal_z = nworld[2];
            
            //unsigned char overlap_bool = (kf->overlap_mask_image_).ptr<unsigned char>(y)[x];
            //if (overlap_bool == 0)
            //{
              //(kf->point_cloud_novel_)->push_back(point_RGB);
            //}
            
            if (!std::isnan(ncam[0]))
            {  
              unsigned char overlap_bool = (kf->overlap_mask_image_).ptr<unsigned char>(y)[x];
              if (overlap_bool == 0)
                (kf->point_cloud_novel_)->push_back(point_RGB); 
                           
              point_cloud_->push_back(point_RGB);
              cloud2image_indices_.push_back(y*cols+x);
              image2cloud_indices_[y*cols + x] = point_id;
              normal_cloud_->push_back(point_normal);
              point_id++;        
            }
          }
          else
          {
            point_cloud_->push_back(point_RGB);
            cloud2image_indices_.push_back(y*cols+x);
            image2cloud_indices_[y*cols + x] = point_id;
            point_id++;        
          }
        }
      }
    }
  }
}


void
RGBID_SLAM::KeyframeManager::getRGBNegentropy (PixelRGB color_scale, std::vector<PixelRGB>& negentropy_RGB)
{
  for (int y = 0; y < negentropy_image_.rows; y++)
  {
    for (int x = 0; x < negentropy_image_.cols; x++)
    {   
      float sat = negentropy_image_.ptr<float>(y)[x];
      int r = (int) ( (1-sat)*((float) color_scale.r) + sat*255.f);
      int g = (int) ( (1-sat)*((float) color_scale.g) + sat*255.f);
      int b = (int) ( (1-sat)*((float) color_scale.b) + sat*255.f);
      PixelRGB new_color;
      new_color.r = r; new_color.g = g; new_color.b = b;
      
      negentropy_RGB[y*negentropy_image_.cols+x] = new_color;      
    }
  }
}

