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
#include "loop_closer.h"
#include "util_funcs.h"
#include "sampler.h"

#include <Eigen/LU>

static Sampler prng;

static bool compareLoopConstraints (RGBID_SLAM::PoseConstraint c_i, RGBID_SLAM::PoseConstraint c_j)
{
  return c_i.information_.determinant() > c_j.information_.determinant();
}

RGBID_SLAM::LoopCluster::LoopCluster (int init_unupdated_time)
{
  unupdated_time_ = init_unupdated_time;
}

void RGBID_SLAM::LoopCluster::sortLoopsByInformation()
{
  std::sort(loop_constraints_.begin(), loop_constraints_.end(), compareLoopConstraints);
}

RGBID_SLAM::LoopCloser::LoopCloser(int min_KF_separation, float norm_score_th /*TODO:arg list*/):min_KF_separation_(min_KF_separation),norm_score_th_(norm_score_th)
{
  RANSAC_confidence_ = 0.999f;
  RANSAC_inlier_ratio_=0.3;
  RANSAC_min_points_ = 3;
  mahalannobis_th_ = 3.94f;
  min_required_inliers_ = 50;
  match_score_ratio_th_ = 0.7f;
  max_loop_separation_ = 20; //loop clusters are matched windows with width lower than max_loop_separation
  enabled_flag_ = 1;
  std::cout << "graph optim" << std::endl;
  //PnP_graph_ptr_.reset(new PnPGraphOptimiser);
  std::cout << "kf align" << std::endl;
  keyframe_align_ptr_.reset(new KeyframeAlign);
  std::cout << "vocab" << std::endl;
  vocabulary_.reset(new VocabularyORB);
}

RGBID_SLAM::LoopCloser::~LoopCloser()
{
  vocabulary_->clear();
  vocabulary_.reset();
}

void 
RGBID_SLAM::LoopCloser::loadSettings(const Settings& settings)
{
  Section loop_closer_section;
  
  if (settings.getSection("LOOP_CLOSER",loop_closer_section))
  {
    std::cout << "LOOP_CLOSER" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    Entry entry;
    
    if (loop_closer_section.getEntry("ENABLED",entry))
    {
      std::string enabled_str = entry.getValue();
      
      enabled_flag_ = atoi(enabled_str.c_str());
      
      std::cout << "  ENABLED: " << enabled_flag_ << std::endl;    
    }
    
    if (loop_closer_section.getEntry("RANSAC_CONFIDENCE",entry))
    {
      std::string RANSAC_confidence_str = entry.getValue();
      
      RANSAC_confidence_ = atof(RANSAC_confidence_str.c_str());
      
      std::cout << "  RANSAC_CONFIDENCE: " << RANSAC_confidence_ << std::endl;    
    }
    
    if (loop_closer_section.getEntry("RANSAC_INLIER_RATIO",entry))
    {
      std::string RANSAC_inlier_ratio_str = entry.getValue();
      
      RANSAC_inlier_ratio_ = atof(RANSAC_inlier_ratio_str.c_str());
      
      std::cout << "  RANSAC_INLIER_RATIO: " << RANSAC_inlier_ratio_ << std::endl;    
    }
    
    if (loop_closer_section.getEntry("RANSAC_MIN_POINTS",entry))
    {
      std::string RANSAC_min_points_str = entry.getValue();
      
      RANSAC_min_points_ = atoi(RANSAC_min_points_str.c_str());
      
      std::cout << "  RANSAC_MIN_POINTS: " << RANSAC_min_points_ << std::endl;    
    }
    
    if (loop_closer_section.getEntry("MAHALANNOBIS_THRESHOLD",entry))
    {
      std::string mahalannobis_th_str = entry.getValue();
      
      mahalannobis_th_ = atof(mahalannobis_th_str.c_str());
      
      std::cout << "  MAHALANNOBIS_THRESHOLD: " << mahalannobis_th_ << std::endl;    
    }
    
    if (loop_closer_section.getEntry("MIN_REQUIRED_INLIERS",entry))
    {
      std::string min_required_inliers_str = entry.getValue();
      
      min_required_inliers_ = atoi(min_required_inliers_str.c_str());
      
      std::cout << "  MIN_REQUIRED_INLIERS: " << min_required_inliers_ << std::endl;    
    }
    
    if (loop_closer_section.getEntry("MIN_KF_SEPARATION",entry))
    {
      std::string min_KF_separation_str = entry.getValue();
      
      min_KF_separation_ = atoi(min_KF_separation_str.c_str());
      
      std::cout << "  MIN_KF_SEPARATION: " << min_KF_separation_ << std::endl;    
    }
    
    if (loop_closer_section.getEntry("NORMALISED_BOW_SCORE_THRESHOLD",entry))
    {
      std::string norm_score_th_str = entry.getValue();
      
      norm_score_th_ = atof(norm_score_th_str.c_str());
      
      std::cout << "  NORMALISED_BOW_SCORE_THRESHOLD: " << norm_score_th_ << std::endl;    
    }
    
    if (loop_closer_section.getEntry("MATCH_SCORE_RATIO_THRESHOLD",entry))
    {
      std::string match_score_ratio_th_str = entry.getValue();
      
      match_score_ratio_th_ = atof(match_score_ratio_th_str.c_str());
      
      std::cout << "  MATCH_SCORE_RATIO_THRESHOLD: " << match_score_ratio_th_ << std::endl;    
    }
    
    std::cout << std::endl;
    std::cout << std::endl;
  }
}  

void
RGBID_SLAM::LoopCloser::loadVocabulary(std::string filename_vocabulary)
{
  vocabulary_->load_ifstream(filename_vocabulary);
}

void
RGBID_SLAM::LoopCloser::computeBowHistograms(KeyframePtr& keyframe_query)
{  
  //keyframe_query->bow_histogram_list_.resize(keyframe_query->bow_histogram_list_size_);
  //keyframe_query->bow_feature_vector_list_.resize(keyframe_query->bow_histogram_list_size_);
  
  //TODO: take many bow histograms after aplying image masks
  vocabulary_->transform(keyframe_query->descriptors_,keyframe_query->bow_histogram_list_[0], keyframe_query->bow_feature_vector_list_[0],4);   
  
  for (int i = 1; i < keyframe_query->bow_histogram_list_size_; i++)
  {
    vocabulary_->transform(keyframe_query->masked_descriptors_[i],keyframe_query->bow_histogram_list_[i], keyframe_query->bow_feature_vector_list_[i],4);  
  }   
}


bool
RGBID_SLAM::LoopCloser::detectLoopClosures(std::vector<KeyframePtr> &keyframe_database, KeyframePtr& keyframe_query, std::vector<LoopCluster> &loop_clusters /*TODO:arg list*/)
{
  //First compute score reference
  //std::cout << keyframe_database.size() << " " << min_KF_separation_ << std::endl;
  if (keyframe_database.size() < std::max(1,min_KF_separation_))
    return false; 
  
  if (!enabled_flag_)
    return false;
    
  DBoW2::BowVector query_hist = keyframe_query->bow_histogram_list_[0];
  DBoW2::BowVector last_hist = keyframe_database.back()->bow_histogram_list_[0];
            
  
  float score_ref = vocabulary_->score(query_hist, last_hist);
  
  //std::cout << "score ref: " << score_ref << std::endl;  
  
  bool candidate_found = false;
  std::multimap<float, std::pair<int,int> > new_cand_loop_matches_by_score;
  std::multimap<int, std::pair<int,int> > new_cand_loop_matches_by_separation;
  
  for (std::vector<KeyframePtr>::const_iterator it_kf = keyframe_database.begin(); 
        it_kf != keyframe_database.end()-min_KF_separation_; it_kf++)
  {
    //Loop through histograms of candidates
    DBoW2::BowVector cand_hist = (*it_kf)->bow_histogram_list_[0];
    float score_cand = vocabulary_->score(query_hist, cand_hist);
    float norm_score = score_cand / score_ref;
        
    if (norm_score > norm_score_th_)
    {
      std::pair<int,int> new_cand_loop_match;
      new_cand_loop_match = std::make_pair(keyframe_query->kf_id_, (*it_kf)->kf_id_);
      int kf_time_separation = std::abs(keyframe_query->kf_id_ - (*it_kf)->kf_id_);
      candidate_found = true;
      new_cand_loop_matches_by_score.insert(std::make_pair(norm_score, new_cand_loop_match));
      new_cand_loop_matches_by_separation.insert(std::make_pair(kf_time_separation, new_cand_loop_match));
    }
  }
  
  std::vector<std::pair<int,int> > new_cand_loop_matches;
  
  if (candidate_found)
  {  
    std::vector<int> added_candidates;
        
    {
      //std::multimap<float, std::pair<int,int> >::iterator it_mm;
      std::multimap<int, std::pair<int,int> >::iterator it_mm;
      
      //it_mm = new_cand_loop_matches_by_score.end();
      it_mm = new_cand_loop_matches_by_separation.end();
      --it_mm;
      
      //int n_selected_loop_matches = std::min(5,(int) new_cand_loop_matches_by_score.size());
      int n_selected_loop_matches = std::min(2,(int) new_cand_loop_matches_by_separation.size());
      
      for (int i=0; i<n_selected_loop_matches; i++)
      {
        new_cand_loop_matches.push_back(it_mm->second);
        --it_mm;
      }
    }
    
    {
      std::multimap<float, std::pair<int,int> >::iterator it_mm;
      it_mm = new_cand_loop_matches_by_score.end();
      --it_mm;
      
      int n_selected_loop_matches = std::min(2,(int) new_cand_loop_matches_by_score.size());
      
      for (int i=0; i<n_selected_loop_matches; i++)
      {
        if (find(new_cand_loop_matches.begin(),new_cand_loop_matches.end(),it_mm->second) == new_cand_loop_matches.end() )
          new_cand_loop_matches.push_back(it_mm->second);
          
        --it_mm;
      }
    }
  }
  
  //std::pair<int,int> seq_loop_match;
  //seq_loop_match = std::make_pair(keyframe_query->kf_id_, (*(keyframe_database.end()-1))->kf_id_);
  //candidate_found = true;
  //new_cand_loop_matches.push_back(seq_loop_match);
  
  bool some_loop_computed = false;
  //std::cout << new_cand_loop_matches.size() << std::endl;
  for (int cand_idx=0; cand_idx < new_cand_loop_matches.size(); cand_idx++)
  {
    std::vector < std::vector<cv::DMatch> > raw_kNNmatches;
    
    int kNN = 2;

    bool cross_check_flag = false;    
    cv::BFMatcher ORBmatcher(cv::NORM_HAMMING, cross_check_flag);  
    cv::Mat desc_query;
    cv::Mat desc_candidate;

    KeyframePtr keyframe_candidate = keyframe_database[new_cand_loop_matches[cand_idx].second];
    
    std::cout << "FOUND CANDIDATE LOOP!! " << keyframe_query->id_ << " " << keyframe_candidate->id_ << std::endl;
    
    descVec2DescMat(keyframe_query->descriptors_, desc_query);
    descVec2DescMat(keyframe_candidate->descriptors_, desc_candidate);
      
    ORBmatcher.knnMatch(desc_query, desc_candidate, raw_kNNmatches, kNN, cv::Mat(), true);
    
    std::vector<cv::DMatch>  raw_matches;
    for (int i=0; i<raw_kNNmatches.size(); i++)
    {
      if (raw_kNNmatches[i][0].distance < match_score_ratio_th_*raw_kNNmatches[i][1].distance)
        raw_matches.push_back(raw_kNNmatches[i][0]);
    }
    
    if (raw_matches.size() < min_required_inliers_)
      continue;
    
    Eigen::Affine3d trafo_query2cand;
    Eigen::Affine3d qTc_ini;
    Eigen::Matrix<double, 6, 6> cov_qTc;
    
    //Apply RANSAC
    if (computeRANSACTrafo3D(keyframe_query, keyframe_candidate, raw_matches, trafo_query2cand,qTc_ini, cov_qTc))    
    { 
      keyframe_align_ptr_->alignKeyframes(keyframe_query, keyframe_candidate, trafo_query2cand, cov_qTc);
      
      RGBID_SLAM::device::sync ();
      
      //cov_qTc = 0.001*cov_qTc;
      
      PoseConstraint loop_constr_new(keyframe_query->id_, keyframe_candidate->id_, PoseConstraint::LC_KF, 
                                                               trafo_query2cand.linear(), trafo_query2cand.translation(), 1.f, cov_qTc); //TODO: cov?
      
      std::vector<LoopCluster>::iterator clu_it;
        
      for (clu_it = loop_clusters.begin(); clu_it != loop_clusters.end(); clu_it++)
      {
        std::vector<PoseConstraint>::iterator c_it;
        //int min_ini_separation = max_loop_separation_;
        //int min_end_separation = max_loop_separation_;
        
        for (c_it = (*clu_it).loop_constraints_.begin(); c_it != (*clu_it).loop_constraints_.end(); c_it++)
        {
          //if  (std::abs((*c_it).ini_id_-loop_constr_new.ini_id_) < min_ini_separation) 
            //min_ini_separation = std::abs((*c_it).ini_id_-loop_constr_new.ini_id_);
            
          //if  (std::abs((*c_it).end_id_-loop_constr_new.end_id_) < min_end_separation)  
            //min_end_separation = std::abs((*c_it).end_id_-loop_constr_new.end_id_);  
          
          
          if ( (std::abs((*c_it).ini_id_-loop_constr_new.ini_id_) > max_loop_separation_) ||
             (std::abs((*c_it).end_id_-loop_constr_new.end_id_) > max_loop_separation_)   )
          {
            break;
          }          
        } 
        
        if (c_it == (*clu_it).loop_constraints_.end())
        {
          (*clu_it).loop_constraints_.push_back(loop_constr_new);
          (*clu_it).unupdated_time_ = 0;
          break;
        }
      }
      
      if (clu_it == loop_clusters.end())
      {
        LoopCluster new_cluster;
        new_cluster.loop_constraints_.push_back(loop_constr_new);
        new_cluster.unupdated_time_ = 0;
        loop_clusters.push_back(new_cluster);
      } 
      
      std::cout << "some loop computed" << std::endl;
      some_loop_computed = true;  
    }  
  }
  
  return some_loop_computed;
}

bool
RGBID_SLAM::LoopCloser::computeRANSACTrafo3D( KeyframePtr keyframe_query,
                                                   KeyframePtr keyframe_candidate,
                                                   std::vector <cv::DMatch> &raw_matches,
                                                   Eigen::Affine3d& qTc, Eigen::Affine3d& qTc_ini,
                                                   Eigen::Matrix<double,6,6>& cov_qTc)
{
  if (raw_matches.size() < RANSAC_min_points_)
    return false;
    
  //pcl::ScopeTime tLC ("compute Loop Closure Trafo");
  int num_iters = ((int) (std::log(1.f - RANSAC_confidence_) / std::log(1-std::pow(RANSAC_inlier_ratio_,(float) RANSAC_min_points_)) -1.f) ) + 1;
  //std::cout << "RANSAC iterations: " << num_iters << std::endl;
  
  std::vector <cv::DMatch> RANSAC_matches;
  std::vector <cv::DMatch> inlier_matches;
  std::vector <cv::DMatch> highest_inlier_matches;
  
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > RANSAC_query_3Dpoints;
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > RANSAC_candidate_3Dpoints;
  
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > best_RANSAC_query_3Dpoints;
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > best_RANSAC_candidate_3Dpoints;
  
  highest_inlier_matches.clear();
  
  prng.reset(0x34985739);
  
  {
    //pcl::ScopeTime tRANSAC ("   compute best RANSAC trafo");
    for (int iter=0; iter<num_iters; iter++)
    {
      RANSAC_matches.clear();
      RANSAC_query_3Dpoints.clear();
      RANSAC_candidate_3Dpoints.clear();
      
      selectRandomMatches(RANSAC_min_points_, raw_matches, RANSAC_matches);
     
      for (int j=0; j<RANSAC_matches.size(); j++)
      {
        RANSAC_query_3Dpoints.push_back(keyframe_query->points3D_[RANSAC_matches[j].queryIdx]);
        RANSAC_candidate_3Dpoints.push_back(keyframe_candidate->points3D_[RANSAC_matches[j].trainIdx]);
      }      
      
      Eigen::Affine3d qTc_trial;
      
      //std::cout << "  computePoseFrom3Dto3DCorrespondences" << std::endl;
      computePoseFrom3Dto3DCorrespondencesClosedForm(RANSAC_query_3Dpoints, RANSAC_candidate_3Dpoints, qTc_trial);
      
      //Vote trafo estimate
      inlier_matches.clear();
      
      //std::cout << "  voteRANSACPose" << std::endl;
      voteRANSACPose(keyframe_query, keyframe_candidate, qTc_trial, raw_matches, inlier_matches);
     
      //std::cout << "  updateBestInlierMatches" << std::endl;
      if (inlier_matches.size() > highest_inlier_matches.size())
      {
        highest_inlier_matches = inlier_matches;
        qTc_ini = qTc_trial;
        best_RANSAC_query_3Dpoints = RANSAC_query_3Dpoints;
        best_RANSAC_candidate_3Dpoints = RANSAC_candidate_3Dpoints;
      }
      //////////////////////////      
    }  
  }
  
  std::cout << "highest inlier matches: " << highest_inlier_matches.size() << std::endl;
  std::cout << "raw matches: " << raw_matches.size() << std::endl;
  
  //For debugging
  //{
    //pcl::ScopeTime tDebug ("    saveDataForDebug");
    //cv::Mat matches_im;
    //cv::Mat matches_im_rgb;  
      
    //cv::drawMatches(keyframe_query->rgb_image_, keyframe_query->keypoints_, 
                    //keyframe_candidate->rgb_image_, keyframe_candidate->keypoints_, 
                    //raw_matches, matches_im);
                    
    //cv::cvtColor(matches_im, matches_im_rgb, CV_BGR2RGB);
    
    //char im_file[128];
    //sprintf(im_file, "debug_folder/matchedFrames%08dand%08d.png", keyframe_query->id_,keyframe_candidate->id_);            
    //cv::imwrite(im_file, matches_im_rgb);
  //}
  ////////////////////////////////////////////////
  
  std::vector<cv::Point2f> inlier_query_cv_points;
  std::vector<cv::Point2f> inlier_candidate_cv_points;
  
  for (int j=0; j<highest_inlier_matches.size(); j++)
  {
    inlier_query_cv_points.push_back(keyframe_query->keypoints_[highest_inlier_matches[j].queryIdx].pt);
    inlier_candidate_cv_points.push_back(keyframe_candidate->keypoints_[highest_inlier_matches[j].trainIdx].pt);
  }
  
  
  
  if ((highest_inlier_matches.size() < min_required_inliers_) )
    return false;
    
  
  double area_ratio_ch_query = computeConvexHullArea(inlier_query_cv_points) / ((double) (keyframe_query->cols_*keyframe_query->rows_));
  double area_ratio_ch_candidate = computeConvexHullArea(inlier_candidate_cv_points) / ((double) (keyframe_candidate->cols_*keyframe_candidate->rows_));
  std::cout << "Area ratios: " << area_ratio_ch_query << " and " << area_ratio_ch_candidate << std::endl;
    
  if ((area_ratio_ch_query < 0.05) || (area_ratio_ch_candidate < 0.05))
    return false;
  //if (highest_inlier_matches.size() < RANSAC_inlier_ratio_*raw_matches.size())
    //return false;
  
  //If there are enough inliers, we have a good match. Proceed with pose refinement by iterative optimisation
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > inlier_query_3Dpoints;
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > inlier_candidate_3Dpoints;
  std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > inlier_query_2Dprojections;
  std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > inlier_candidate_2Dprojections;
  std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > inlier_query_3Dcovs;
  std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > inlier_candidate_3Dcovs;
  std::vector<Eigen::Matrix2d,Eigen::aligned_allocator<Eigen::Matrix2d> > inlier_candidate_2Dcovs;  
  
  
  for (int j=0; j<highest_inlier_matches.size(); j++)
  {
    Eigen::Vector3d X_q = keyframe_query->points3D_[highest_inlier_matches[j].queryIdx];
    inlier_query_3Dpoints.push_back(X_q);
    
    Eigen::Vector2d p_q = keyframe_query->points2D_[highest_inlier_matches[j].queryIdx];
    inlier_query_2Dprojections.push_back(p_q);
    
    Eigen::Matrix3d cov_Xq = keyframe_query->points3D_cov_[highest_inlier_matches[j].queryIdx];
    inlier_query_3Dcovs.push_back(cov_Xq); 
    
    Eigen::Vector2d p_c = keyframe_candidate->points2D_[highest_inlier_matches[j].trainIdx];    
    inlier_candidate_2Dprojections.push_back(p_c);
    
    Eigen::Matrix2d cov_u = keyframe_candidate->points2D_cov_[highest_inlier_matches[j].trainIdx];      
    inlier_candidate_2Dcovs.push_back(cov_u);    
    
    Eigen::Vector3d X_c = keyframe_candidate->points3D_[highest_inlier_matches[j].trainIdx];
    inlier_candidate_3Dpoints.push_back(X_c);
    
    Eigen::Matrix3d cov_Xc = keyframe_candidate->points3D_cov_[highest_inlier_matches[j].trainIdx];
    inlier_candidate_3Dcovs.push_back(cov_Xc); 
  }  
  
  double maxdist_q = 0;
  double maxdist_c = 0;
  
  //Check if matches are well spread enough in both keyframes
  for (int i=0; i<inlier_query_2Dprojections.size(); i++)
  {
    Eigen::Vector2d p_qi = inlier_query_2Dprojections[i];
    Eigen::Vector2d p_ci = inlier_candidate_2Dprojections[i];
    
    for (int j=i+1; j<inlier_query_2Dprojections.size(); j++)
    {
      Eigen::Vector2d p_qj = inlier_query_2Dprojections[j];
      Eigen::Vector2d p_cj = inlier_candidate_2Dprojections[j];
      
      Eigen::Vector2d delta_qij = p_qi - p_qj;
      Eigen::Vector2d delta_cij = p_ci - p_cj;
      
      float dist_q = delta_qij.norm();
      float dist_c = delta_cij.norm();   
      
      maxdist_q = (dist_q > maxdist_q) ? dist_q : maxdist_q;   
      maxdist_c = (dist_c > maxdist_c) ? dist_c : maxdist_c;  
    }
  }
  
  qTc = qTc_ini;
  //computePoseWithPnPCorrespondences(keyframe_candidate->K_, inlier_query_3Dpoints, inlier_candidate_2Dprojections, qTc_ini, qTc, cov_qTc, inlier_candidate_2Dcovs, inlier_query_3Dcovs);
  //computePoseWith3D3DCorrespondences(inlier_query_3Dpoints, inlier_candidate_3Dpoints, qTc_ini, qTc, cov_qTc, inlier_query_3Dcovs, inlier_candidate_3Dcovs);
  //computePoseFrom3Dto3DCorrespondencesClosedForm(inlier_query_3Dpoints, inlier_candidate_3Dpoints, qTc);
  
  return true;
}

void 
RGBID_SLAM::LoopCloser::selectRandomMatches(int RANSAC_min_points, std::vector < cv::DMatch > raw_matches, std::vector <cv::DMatch>  &RANSAC_matches)
{
  int selectionable_size = raw_matches.size();
  
  //std::cout << selectionable_size << std::endl;
  
  if (selectionable_size < RANSAC_min_points)
  {
    
    std::cout << "Not enough points for RANSAC" << std::endl;
    return;
  }
  
  for (int i=0; i<RANSAC_min_points; i++)
  {
    //std::cout <<"selectionable size: " << selectionable_size << std::endl;
    double u = prng.rand_uniform01();
    //std::cout <<"u: " << u << std::endl;
    int match_idx = (int) ( u*((double) selectionable_size) );
    //std::cout <<"match idx: " << match_idx << std::endl;
    cv::DMatch temp = raw_matches[match_idx];
    RANSAC_matches.push_back(temp);
    raw_matches[match_idx] = raw_matches[selectionable_size - 1];
    selectionable_size--;    
  }
}
                
//void 
//RGBID_SLAM::LoopCloser::computePoseWithPnPCorrespondences  (Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &points3D_query,
                                                                  //const std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > &projections2D_candidate, 
                                                                  //const Eigen::Affine3d& qTc_init,
                                                                  //Eigen::Affine3d& qTc_optim,
                                                                  //Eigen::Matrix<double,6,6>& cov_qTc,
                                                                  //const std::vector<Eigen::Matrix2d,Eigen::aligned_allocator<Eigen::Matrix2d> > &cov2D_candidate,
                                                                  //const std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > &cov3D_query)  
//{
  //PnP_graph_ptr_->buildGraph3Dto2D(K, points3D_query, projections2D_candidate, qTc_init, cov3D_query, cov2D_candidate);
  //PnP_graph_ptr_->optimiseGraphAndGetPnPTrafo(qTc_optim, cov_qTc);
//} 

//void 
//RGBID_SLAM::LoopCloser::computePoseWith3D3DCorrespondences  (const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &points3D_query,
                                                                  //const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &points3D_candidate, 
                                                                  //const Eigen::Affine3d& qTc_init,
                                                                  //Eigen::Affine3d& qTc_optim,
                                                                  //Eigen::Matrix<double,6,6>& cov_qTc,
                                                                  //const std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > &cov3D_query,
                                                                  //const std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > &cov3D_candidate)  
//{
  //PnP_graph_ptr_->buildGraph3Dto3D(points3D_query, points3D_candidate, qTc_init, cov3D_query, cov3D_candidate);
  //PnP_graph_ptr_->optimiseGraphAndGetPnPTrafo(qTc_optim, cov_qTc);
//} 
                                               
void 
RGBID_SLAM::LoopCloser::computePoseFrom3Dto3DCorrespondencesClosedForm( std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &query_3Dpoints,
                                                                   std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > &candidate_3Dpoints,
                                                                   Eigen::Affine3d& trafo_query2cand)
{
  std::vector<double> weights(query_3Dpoints.size(),1.f);
  
  //First of all we compute the centroids
  Eigen::Vector3d query_3Dcentroid = computeWeightedMean<Eigen::Vector3d>(query_3Dpoints,weights);
  Eigen::Vector3d candidate_3Dcentroid = computeWeightedMean<Eigen::Vector3d>(candidate_3Dpoints,weights);
  
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > query_delta3Dpoints(query_3Dpoints.size());
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > candidate_delta3Dpoints(candidate_3Dpoints.size());
  
  //std::cout << "    computing correlation" << std::endl;
  for (int i=0; i<query_delta3Dpoints.size(); i++)
  {
    query_delta3Dpoints[i] = query_3Dpoints[i] - query_3Dcentroid;
    candidate_delta3Dpoints[i] = candidate_3Dpoints[i] - candidate_3Dcentroid;
  }
  
  Eigen::Matrix3d corr_query2cand = computeWeightedCorrelation<Eigen::Vector3d, Eigen::Matrix3d>(query_delta3Dpoints, candidate_delta3Dpoints, weights);
  
  //std::cout << "    computing trafo" << std::endl;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(corr_query2cand, Eigen::ComputeFullV | Eigen::ComputeFullU);
  
  trafo_query2cand.linear() = svd.matrixU()*svd.matrixV().transpose(); 
  
  trafo_query2cand.translation() = query_3Dcentroid - trafo_query2cand.linear()*candidate_3Dcentroid;  
}
                      
                      
void 
RGBID_SLAM::LoopCloser::voteRANSACPose(KeyframePtr keyframe_query, KeyframePtr keyframe_candidate,
                                            Eigen::Affine3d qTc, std::vector < cv::DMatch > &matches, std::vector < cv::DMatch > &inlier_matches)
{  
  inlier_matches.clear();
  inlier_matches.reserve(matches.size());
    
  Eigen::Matrix3d qRc = qTc.linear();
  Eigen::Vector3d t_qc = qTc.translation();
  
  Eigen::Matrix3d cRq = qRc.inverse();
  Eigen::Vector3d t_cq = -cRq*t_qc;
  
  for (int k=0; k<matches.size() ; k++) 
  {
    Eigen::Vector3d X_q = keyframe_query->points3D_[matches[k].queryIdx];
    Eigen::Vector3d X_c = keyframe_candidate->points3D_[matches[k].trainIdx];
    
    Eigen::Matrix3d cov_Xq = keyframe_query->points3D_cov_[matches[k].queryIdx];
    Eigen::Matrix3d cov_Xc = keyframe_candidate->points3D_cov_[matches[k].trainIdx];
    
    double error_c2q3D = computeNormalisedError3D(X_c, cov_Xc, X_q, cov_Xq, qRc, t_qc);
    double error_q2c3D = computeNormalisedError3D(X_q, cov_Xq, X_c, cov_Xc, cRq, t_cq);
    
    if ((error_c2q3D < mahalannobis_th_) && (error_q2c3D < mahalannobis_th_))// && (error_c2q3D < 7.81) && (error_q2c3D < 7.81) )
    {
      inlier_matches.push_back(matches[k]);
    }
  }
}   
    
    
double
RGBID_SLAM::LoopCloser::computeNormalisedError(KeyframePtr keyframe_a, KeyframePtr keyframe_b,
                                                    Eigen::Vector3d X_a, Eigen::Matrix3d cov_Xa, 
                                                    Eigen::Vector2d p_b, Eigen::Matrix2d cov_pb, 
                                                    Eigen::Matrix3d bRa, Eigen::Vector3d t_ba)
{
  //TODO: Put this into a function for projection error
  Eigen::Vector3d Xb_from_a = bRa*X_a + t_ba;
  Eigen::Vector2d pb_from_a;
  keyframe_b->projectPoint(pb_from_a, Xb_from_a);
  
  Eigen::Matrix<double,3,3> cov_Xb_from_Xa = bRa*cov_Xa*bRa.transpose();
  
  Eigen::Matrix<double,2,3> projJX_b;
  keyframe_b->projectionJacobianAtPoint(Xb_from_a, projJX_b );
  
  Eigen::Matrix<double,2,2> cov_proj2pb = projJX_b*cov_Xb_from_Xa*projJX_b.transpose() + cov_pb;
  
  //Eigen::Matrix<float,2,2> cov_proj2pb = cov_u_and_invdepth.block<2,2>(0,0);
  double err_a2b_mahalannobis = std::sqrt((pb_from_a.transpose()-p_b.transpose())*cov_proj2pb.inverse()*(pb_from_a - p_b));
  
  return err_a2b_mahalannobis;
  //////////////////
}  

double 
RGBID_SLAM::LoopCloser::computeNormalisedError3D(Eigen::Vector3d X_a, Eigen::Matrix3d cov_Xa, 
                                                      Eigen::Vector3d X_b, Eigen::Matrix3d cov_Xb, 
                                                      Eigen::Matrix3d bRa, Eigen::Vector3d t_ba)
{
  //TODO: Put this into a function for projection error
  Eigen::Vector3d Xb_from_a = bRa*X_a + t_ba;
  
  Eigen::Matrix<double,3,3> cov_Xb_from_Xa = bRa*cov_Xa*bRa.transpose();
  Eigen::Matrix<double,3,3> cov_Xb_total = cov_Xb_from_Xa + cov_Xb;
  
  double err_a2b_mahalannobis = std::sqrt((Xb_from_a.transpose()-X_b.transpose())*cov_Xb_total.inverse()*(Xb_from_a - X_b));
  
  return err_a2b_mahalannobis;
  //////////////////
}  
    
  
  


