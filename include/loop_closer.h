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

#ifndef LOOPCLOSER_HPP_
#define LOOPCLOSER_HPP_


#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <set>

#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

#include "types.h"
#include "keyframe.h"
#include "pose_graph_manager.h"
#include "settings.h"
#include "keyframe_align.h"

//#include "PnP_manager.h"


namespace RGBID_SLAM
{  
  typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>  VocabularyORB;
  typedef boost::shared_ptr<VocabularyORB> VocabularyORBPtr;
  
  
  struct LoopCluster
  {
    LoopCluster (int init_unupdated_time = 0);
    
    void sortLoopsByInformation();
    
    std::vector<PoseConstraint> loop_constraints_;
    int unupdated_time_;  
  };
  
  class LoopCloser
  {
    public:
    
      LoopCloser(int min_KF_separation = 3, float norm_score_th = 0.8f/*TODO:arg list*/);  
      
      ~LoopCloser();  
      
      void loadSettings(const Settings& settings);    
      
      void loadVocabulary(std::string filename_vocabulary="../../data/ORBvoc.yml");        
      
      void computeBowHistograms(KeyframePtr& keyframe_query);
      
      bool detectLoopClosures(std::vector<KeyframePtr> &keyframe_database, KeyframePtr& keyframe_query, std::vector<LoopCluster> &loop_clusters);
            
      void computeLoopConstraint(KeyframePtr& keyframe_query, KeyframePtr& keyframe_matched);
        
      bool computeRANSACTrafo3D( KeyframePtr keyframe_query,
                                 KeyframePtr keyframe_candidate,
                                 std::vector <cv::DMatch> &raw_matches,
                                 Eigen::Affine3d& qTc, Eigen::Affine3d& qTc_ini,
                                 Eigen::Matrix<double,6,6>& cov_qTc);
                               
      void computePoseFrom3Dto3DCorrespondencesClosedForm( std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &query_3Dpoints,
                                                                 std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &candidate_3Dpoints,
                                                                 Eigen::Affine3d& trafo_query2cand);
                                                                 
      void computePoseWithPnPCorrespondences  (Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &points3D,
                                                                const std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > &projections2D, 
                                                                const Eigen::Affine3d& pose_init,
                                                                Eigen::Affine3d& pose_optim,                                                                  
                                                                Eigen::Matrix<double,6,6>& cov_optim,
                                                                const std::vector<Eigen::Matrix2d,Eigen::aligned_allocator<Eigen::Matrix2d> > &cov2D,
                                                                const std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > &cov3D);
                                                                
      void computePoseWith3D3DCorrespondences  (const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &points3D_query,
                                                                const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &points3D_candidate, 
                                                                const Eigen::Affine3d& qTc_init,
                                                                Eigen::Affine3d& qTc_optim,
                                                                Eigen::Matrix<double,6,6>& cov_qTc,
                                                                const std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > &cov3D_query,
                                                                const std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > &cov3D_candidate);
                                                                                                                           
      void voteRANSACPose(KeyframePtr keyframe_query, KeyframePtr keyframe_candidate,
                          Eigen::Affine3d qTc, std::vector < cv::DMatch > &matches, std::vector < cv::DMatch > &inlier_matches);
      
      void selectRandomMatches(int RANSAC_min_points, std::vector<cv::DMatch> raw_matches, std::vector<cv::DMatch> &RANSAC_matches);
      
      double computeNormalisedError(KeyframePtr keyframe_a, KeyframePtr keyframe_b,
                                                  Eigen::Vector3d X_a, Eigen::Matrix3d cov_Xa, 
                                                  Eigen::Vector2d p_b, Eigen::Matrix2d cov_pb, 
                                                  Eigen::Matrix3d bRa, Eigen::Vector3d t_ba);
                                                  
      double computeNormalisedError3D(Eigen::Vector3d X_a, Eigen::Matrix3d cov_Xa, 
                                                    Eigen::Vector3d X_b, Eigen::Matrix3d cov_Xb, 
                                                    Eigen::Matrix3d bRa, Eigen::Vector3d t_ba);
      
      VocabularyORBPtr vocabulary_;
      
      char log_file[128];
      KeyframeAlignPtr keyframe_align_ptr_;
      
    private:
    
      int min_KF_separation_;
      float norm_score_th_;
      float RANSAC_confidence_;
      float RANSAC_inlier_ratio_;
      int RANSAC_min_points_;
      float mahalannobis_th_;
      int min_required_inliers_;
      float match_score_ratio_th_;
      int max_loop_separation_;     
      bool enabled_flag_; 
      
      //PnPGraphOptimiserPtr PnP_graph_ptr_;
  };
  
}

#endif /* LOOPCLOSER_HPP_ */
