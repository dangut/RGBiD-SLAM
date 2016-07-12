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

#ifndef POSEGRAPHMANAGER_HPP_
#define POSEGRAPHMANAGER_HPP_

//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>
//#include <pcl/io/ply_io.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>
#include <vector>
#include <deque>
#include <boost/thread/thread.hpp>

#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>

#include <g2o/types/pose_graph/trafo_so3r3.h>
#include <g2o/types/pose_graph/types_six_dof_pose.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/core/batch_stats.h>

#include "keyframe.h"
#include "settings.h"

#include "types.h"

namespace RGBID_SLAM
{   
        
  struct PoseConstraint
  {
    enum {SEQ_ODO, SEQ_KF, LC_KF};
    
    PoseConstraint(int ini_id, int end_id, 
                    int type = SEQ_ODO,
                    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity(),
                    Eigen::Vector3d translation = Eigen::Vector3d::Zero(),
                    float scale = 1.f,
                    Eigen::Matrix<double,6,6> covariance = Eigen::Matrix<double,6,6>::Identity()):
                    ini_id_(ini_id),end_id_(end_id),type_(type),
                    rotation_(rotation),translation_(translation),scale_(scale)
    {
      information_ = covariance.inverse();
    }
    
    int ini_id_;
    int end_id_;
    int type_;  
    Eigen::Matrix3d rotation_;
    Eigen::Vector3d translation_;
    Eigen::Matrix<double,6,6> information_;
    float scale_;      
    //float scale_info_;
    
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  
  struct Pose
  {
    Pose(int id, 
         Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity(),
         Eigen::Vector3d translation = Eigen::Vector3d::Zero(),
         float scale = 1.f):
         id_(id),rotation_(rotation),translation_(translation),scale_(scale) {};
         
    int id_;
    Eigen::Matrix3d rotation_;
    Eigen::Vector3d translation_;
    float scale_;  
    
    inline Eigen::Affine3d 
    getAffine() const
    {
      Eigen::Affine3d aff;
      aff.linear() = scale_*rotation_;
      aff.translation() = translation_;
      return aff;        
    }
    
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  
      
  class PoseGraph
  {
    public:

      PoseGraph(/*TODO: arguments*/);
      ~PoseGraph(/*TODO: arguments*/);
      
      void loadSettings(const Settings& settings);    
            
      void buildGraph(const std::vector<Pose> &poses, const std::vector<PoseConstraint> &constraints);      
      void optimiseGraph();  
      void optimiseGraphMultilevel();   
      void optimiseGraphSinglelevel();  
      void updatePosesAndKeyframes(std::vector<Pose> &poses, std::vector<KeyframePtr> &keyframes);    
      
      int min_pose_id_;
      int max_pose_id_;
      
            
    private:  
    
      typedef g2o::BlockSolver< g2o::BlockSolverTraits<-1, -1> >  LoopBlockSolver;
       
      typedef g2o::LinearSolverEigen<LoopBlockSolver::PoseMatrixType> LoopLinearSolver;       
        
      //typedef g2o::OptimizationAlgorithmLevenberg OptimAlg;
      typedef g2o::OptimizationAlgorithmGaussNewton OptimAlg;
      //typedef g2o::OptimizationAlgorithmDogleg OptimAlg;
      
      typedef LoopBlockSolver* LoopBlockSolverPtr;
      typedef LoopLinearSolver* LoopLinearSolverPtr;
      typedef OptimAlg* OptimAlgPtr;
      
      g2o::SparseOptimizer optimizer_;	
      LoopLinearSolverPtr linearSolver_ ;
      LoopBlockSolverPtr blockSolver_ ;
      OptimAlgPtr algSolver_;
      
      bool multilevel_optim_flag_;
      bool fix_last_flag_;
      
      g2o::BatchStatisticsContainer  batch_stats_;
      
      Eigen::Matrix3d rotation_last_b4optim_;
      Eigen::Vector3d translation_last_b4optim_;
      int idx_last_b4optim_;
      
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
  
}

#endif /* POSEGRAPHMANAGER_HPP_ */
