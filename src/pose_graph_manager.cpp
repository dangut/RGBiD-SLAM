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


#include "pose_graph_manager.h"
#include <g2o/core/robust_kernel_impl.h>

RGBID_SLAM::PoseGraph::PoseGraph()
{
  linearSolver_ = (new LoopLinearSolver);
  blockSolver_ = (new LoopBlockSolver(linearSolver_));
  algSolver_ = (new OptimAlg(blockSolver_));
  
  //algSolver_->setUserLambdaInit(1e-14);
  optimizer_.setAlgorithm(algSolver_);
  optimizer_.setVerbose(true);
  
  fix_last_flag_ = false;
  
  multilevel_optim_flag_ = true;
}

RGBID_SLAM::PoseGraph::~PoseGraph()
{
  std::cout << "  reset optimizer" << std::endl;
  optimizer_.clear();
}

void 
RGBID_SLAM::PoseGraph::loadSettings(const Settings& settings)
{
  Section pose_graph_section;
  
  if (settings.getSection("POSE_GRAPH", pose_graph_section))
  {
    std::cout << "POSE_GRAPH" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    Entry entry;
    
    if (pose_graph_section.getEntry("MULTILEVEL_OPTIM",entry))
    {
      std::string multilevel_optim_str = entry.getValue();
      
      if ((multilevel_optim_str.compare("TRUE") == 0) || (multilevel_optim_str.compare("1") == 0) )  //enum int type defined in internal.h
		    multilevel_optim_flag_ = true;
      if ((multilevel_optim_str.compare("FALSE") == 0) || (multilevel_optim_str.compare("0") == 0) )  //enum int type defined in internal.h
		    multilevel_optim_flag_ = false;
        
      std::cout << "  MULTILEVEL_OPTI: " << multilevel_optim_str <<  std::endl;
    }
    
    std::cout << std::endl;
    std::cout << std::endl;
  }  
}  

void
RGBID_SLAM::PoseGraph::buildGraph(const std::vector<Pose> &poses, const std::vector<PoseConstraint> &constraints)   
{
  optimizer_.clear();
  
  //COPY VERTICES  
  //First vertices from global poses
  for (std::vector<Pose>::const_iterator p_it = poses.begin();
       p_it != poses.end(); p_it++)
  {    
    g2o::TrafoSO3R3 trafo((*p_it).rotation_.cast<double>(), (*p_it).translation_.cast<double>());
    
    //Debugging
    for (int i=0; i<7; i++)
    {
      if (std::isnan(trafo[i]))
      {
        std::cout << "Pose " << (*p_it).id_ << " is a nan " << std::endl;
        break;
      }
    }
    
    g2o::VertexSO3R3* vSE3 = new g2o::VertexSO3R3();
    vSE3->setEstimate(trafo);
    vSE3->setId((*p_it).id_);
    //We fix last pose instead of first to ensure continuity in visualised trajectory when appending new unoptimised poses from visodo
    
    if (fix_last_flag_)
      vSE3->setFixed(p_it == (poses.end()-1));
    else
      vSE3->setFixed(p_it == (poses.begin()));
      
    optimizer_.addVertex(vSE3);
  }
  
  int max_kf_id = 0;
  int min_kf_lc_id = std::numeric_limits<int>::max();
  double accumLC_error = 0.0;
  //COPY CONSTRAINTS
  for (std::vector<PoseConstraint>::const_iterator c_it = constraints.begin();
       c_it != constraints.end(); c_it++)
  {      
    int from = (*c_it).ini_id_;
    int to = (*c_it).end_id_;    
    
    g2o::EdgeSO3R3* eSE3 = new g2o::EdgeSO3R3();
    eSE3->vertices()[0] = optimizer_.vertex(from);
    eSE3->vertices()[1] = optimizer_.vertex(to);
    
    g2o::TrafoSO3R3 trafo((*c_it).rotation_.cast<double>(), (*c_it).translation_.cast<double>());
    Eigen::Matrix<double, 6, 6> info_mat = (*c_it).information_;
    
    
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6> > es(info_mat);    
    
    eSE3->setMeasurement(trafo);
    
    eSE3->setInformation((*c_it).information_);     
    
    eSE3->computeError();
    
    //g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    //eSE3->setRobustKernel(rk);
    //eSE3->robustKernel()->setDelta(1.345);
    
    if ((*c_it).type_ == PoseConstraint::SEQ_ODO)
    {
      eSE3->setLevel(1);
    }
    else if ( ((*c_it).type_ == PoseConstraint::SEQ_KF) || ((*c_it).type_ == PoseConstraint::LC_KF) )
    {
      eSE3->setLevel(2);
      
      max_kf_id = std::max(max_kf_id, std::max(from,to));
      
      if ((*c_it).type_ == PoseConstraint::LC_KF)
      {
        min_kf_lc_id = std::min(min_kf_lc_id, std::min(from,to));
      }
    }
    
    optimizer_.addEdge(eSE3);    
  }
  
  if (fix_last_flag_ && max_kf_id > 0)
    optimizer_.vertex(max_kf_id)->setFixed(true);
  else if(!fix_last_flag_ && min_kf_lc_id < std::numeric_limits<int>::max())
    optimizer_.vertex(min_kf_lc_id)->setFixed(true);
  
  rotation_last_b4optim_ = poses.back().rotation_;
  translation_last_b4optim_ = poses.back().translation_;
  idx_last_b4optim_ = poses.size() -1;
  
  
}

void
RGBID_SLAM::PoseGraph::optimiseGraph()
{
  if (multilevel_optim_flag_ == true)
    optimiseGraphMultilevel();
  else
    optimiseGraphSinglelevel();
}
  
void
RGBID_SLAM::PoseGraph::optimiseGraphMultilevel()
{
  //Optimise in level 2
  std::cout << "optimize level 2" << std::endl;
  optimizer_.initializeOptimization(2);
  //std::cout << "iteration=-1     chi2= " << optimizer_.activeChi2() << std::endl;
  optimizer_.optimize(10);
  
  //Optimise in level 1 fixing those vertices optimised in level 2
  for (g2o::OptimizableGraph::VertexContainer::const_iterator v_it = optimizer_.activeVertices().begin(); 
       v_it != optimizer_.activeVertices().end(); v_it++)
  {
    (*v_it)->setFixed(true);
  }
  
  std::cout << "optimize level 1" << std::endl;
  optimizer_.initializeOptimization(1);
  //std::cout << "iteration=-1     chi2= " << optimizer_.activeChi2() << std::endl;
  optimizer_.optimize(5);
}

void
RGBID_SLAM::PoseGraph::optimiseGraphSinglelevel()
{
  //Optimise in all levels
  optimizer_.initializeOptimization(-1); //negative level->use edges in all levels
  //std::cout << "iteration=-1     chi2= " << optimizer_.activeChi2() << std::endl;
  optimizer_.optimize(10);
}

void
RGBID_SLAM::PoseGraph::updatePosesAndKeyframes(std::vector<Pose> &poses, std::vector<KeyframePtr> &keyframes)    
{
  Eigen::Matrix3d rotation_last_afteroptim = poses[idx_last_b4optim_].rotation_;
  Eigen::Vector3d translation_last_afteroptim = poses[idx_last_b4optim_].translation_;
  
  for (std::vector<Pose>::iterator p_it = poses.begin();
       p_it != poses.end(); p_it++)
  {
    g2o::VertexSO3R3* vSE3 = dynamic_cast<g2o::VertexSO3R3*>(optimizer_.vertex((*p_it).id_));
    if (vSE3 != 0)
    {
      (*p_it).rotation_ = vSE3->estimate().rotation();
      (*p_it).translation_ = vSE3->estimate().translation();  
    }
    else //pose was added by viodo while optim
    {
      Eigen::Matrix3d delta_rotation = rotation_last_b4optim_.transpose()*(*p_it).rotation_;
      Eigen::Vector3d delta_translation = rotation_last_b4optim_.transpose()*((*p_it).translation_ - translation_last_b4optim_);
      (*p_it).rotation_ = rotation_last_afteroptim*delta_rotation;
      (*p_it).translation_ = translation_last_afteroptim + rotation_last_afteroptim*delta_translation;
    }
  }  
  
  for (std::vector<KeyframePtr>::iterator kf_it = keyframes.begin();
       kf_it != keyframes.end(); kf_it++)
  {
    g2o::VertexSO3R3* vSE3 = dynamic_cast<g2o::VertexSO3R3*>(optimizer_.vertex((*kf_it)->id_));
    if (vSE3 != 0)
    {
      (*kf_it)->setPose(vSE3->estimate().rotation(),vSE3->estimate().translation());
      //(*kf_it)->computeAlignedPointCloud((*kf_it)->pose_);
    }
  } 
}
