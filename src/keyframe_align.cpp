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
#include "keyframe_align.h"
#include "util_funcs.h"

//#include "internal.h"
using namespace RGBID_SLAM::device;
using RGBID_SLAM::device::device_cast;



RGBID_SLAM::KeyframeAlign::KeyframeAlign()
{
  int cols = 640;
  int rows = 480;
  
  numSMs_ = 5;
  
  aligned_pose_.linear() = Eigen::Matrix3d::Identity();
  aligned_pose_.translation() = Eigen::Vector3d(0,0,0);
  
  const int iters[] = {5, 5, 3, 0, 0, 0};
  finest_level_ = 0;
  
  std::copy (iters, iters + LEVELS, alignment_iterations_);
  
  depthinvs_ini_.resize(LEVELS);
  warped_depthinvs_end_.resize(LEVELS);
  xGradsDepthinv_ini_.resize(LEVELS);
  yGradsDepthinv_ini_.resize(LEVELS);
  res_depthinvs_.resize(LEVELS);
  vertices_ini_.resize(LEVELS);
  normals_ini_.resize(LEVELS);
  
  intensities_ini_.resize(LEVELS);
  intensities_end_.resize(LEVELS);
  warped_intensities_end_.resize(LEVELS);
  xGradsIntensity_ini_.resize(LEVELS);
  yGradsIntensity_ini_.resize(LEVELS);
  res_intensities_.resize(LEVELS);
  
  depthinvs_end_.resize(LEVELS);
  projected_transformed_points_.resize(LEVELS);
  
  depthinv_warped_in_end_.create(rows,cols);
  warped_weight_end_.create(rows,cols);
  
  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    intensities_ini_[i].create (pyr_rows, pyr_cols);
    intensities_end_[i].create (pyr_rows, pyr_cols);
    warped_intensities_end_[i].create (pyr_rows, pyr_cols);
    
    depthinvs_ini_[i].create (pyr_rows, pyr_cols);
    depthinvs_end_[i].create (pyr_rows, pyr_cols);
    warped_depthinvs_end_[i].create (pyr_rows, pyr_cols);
    
    vertices_ini_[i].create (3*pyr_rows, pyr_cols); 
    normals_ini_[i].create (3*pyr_rows, pyr_cols); 
    
    xGradsDepthinv_ini_[i].create (pyr_rows, pyr_cols);
    yGradsDepthinv_ini_[i].create (pyr_rows, pyr_cols);
    res_depthinvs_[i].create(pyr_rows * pyr_cols);	
    
    xGradsIntensity_ini_[i].create (pyr_rows, pyr_cols);
    yGradsIntensity_ini_[i].create (pyr_rows, pyr_cols);
    res_intensities_[i].create(pyr_rows * pyr_cols);
    
    projected_transformed_points_[i].create(2*pyr_rows, pyr_cols);
  }
}

RGBID_SLAM::KeyframeAlign::~KeyframeAlign()
{}

bool
RGBID_SLAM::KeyframeAlign::alignKeyframes(KeyframePtr& kf_ini, KeyframePtr& kf_end, Eigen::Affine3d& pose_ini2end, Eigen::Matrix<double,6,6>& covariance_ini2end)
{
  Eigen::Matrix3d rotation_ini2end = pose_ini2end.linear();
  Eigen::Vector3d translation_ini2end = pose_ini2end.translation();
  
  alignKeyframes(kf_ini, kf_end, rotation_ini2end, translation_ini2end, covariance_ini2end);
  
  pose_ini2end.linear() = rotation_ini2end;
  pose_ini2end.translation() = translation_ini2end; 
  
  return true; 
}

bool
RGBID_SLAM::KeyframeAlign::alignKeyframes(KeyframePtr& kf_ini, KeyframePtr& kf_end, Eigen::Matrix3d& rotation_ini2end, Eigen::Vector3d& translation_ini2end, Eigen::Matrix<double,6,6>& covariance_ini2end)
{
  //Eigen::Affine3f pose_ini2end = (kf_ini->pose_.inverse())*(kf_end->pose_);
  Intr cam_intrinsics (kf_ini->K_(0,0), kf_ini->K_(1,1), kf_ini->K_(0,2), kf_ini->K_(1,2), 0.075);
  depthinvs_ini_[0].upload(kf_ini->depthinv_image_.ptr<float>(), kf_ini->depthinv_image_.cols*kf_ini->depthinv_image_.elemSize(), kf_ini->depthinv_image_.rows, kf_ini->depthinv_image_.cols);   
  depthinvs_end_[0].upload(kf_end->depthinv_image_.ptr<float>(), kf_end->depthinv_image_.cols*kf_end->depthinv_image_.elemSize(), kf_end->depthinv_image_.rows, kf_end->depthinv_image_.cols); 
  
  cv::Mat intensity_float_ini, intensity_float_end;
  
  kf_ini->grey_image_.convertTo(intensity_float_ini,CV_32F);
  kf_end->grey_image_.convertTo(intensity_float_end,CV_32F);
  
  intensities_ini_[0].upload(intensity_float_ini.ptr<float>(), intensity_float_ini.cols*intensity_float_ini.elemSize(), intensity_float_ini.rows, intensity_float_ini.cols);   
  intensities_end_[0].upload(intensity_float_end.ptr<float>(), intensity_float_end.cols*intensity_float_end.elemSize(), intensity_float_end.rows, intensity_float_end.cols); 
  
  
  float sigma_depthinv;
  float bias_depthinv;     
  float sigma_intensity;
  float bias_intensity;   
  float nu_depthinv;
  float nu_intensity;
  //float sigma_int_ref, sigma_depthinv_ref;
  //sigma_int_ref = 5.f;
  //sigma_depthinv_ref = 0.0025f;		
  
  float chi_test = 1.f;  
  //float chi_square_prev = 1.f;
  float chi_square = 1.f;
  float Ndof = 640.f*480.f;
  float Ndof_prev = 640.f*480.f;
  float RMSE = 9999.f;
  float RMSE_prev = 9999.f;  
  //bool last_iter_flag = false;   
  
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A_total;
  Eigen::Matrix<double, 6, 1> b_total; 
  
  Eigen::Matrix3d current_rotation = rotation_ini2end;
  Eigen::Vector3d current_translation = translation_ini2end;
  
  for (int i = 1; i < LEVELS; ++i)
  {
    pyrDownDepth (depthinvs_ini_[i-1], depthinvs_ini_[i], numSMs_);
    pyrDownDepth (depthinvs_end_[i-1], depthinvs_end_[i], numSMs_);
    
    pyrDownIntensity (intensities_ini_[i-1], intensities_ini_[i], numSMs_);
    pyrDownIntensity (intensities_end_[i-1], intensities_end_[i], numSMs_);
  }   
  ////////////////////////////////
  
  
  for (int i = 0; i < LEVELS; ++i)
  {     
    computeGradientDepth(depthinvs_ini_[i], xGradsDepthinv_ini_[i], yGradsDepthinv_ini_[i], numSMs_); 
    computeGradientIntensity(intensities_ini_[i], xGradsIntensity_ini_[i], yGradsIntensity_ini_[i], numSMs_); 
    
    //createVMap( cam_intrinsics(i), depthinvs_ini_[i], vertices_ini_[i], numSMs_); 
    //createNMap( vertices_ini_[i], normals_ini_[i], numSMs_); 
    //createNMapGradients(cam_intrinsics(i), depthinvs_ini_[i], xGradsDepthinv_ini_[i], yGradsDepthinv_ini_[i], normals_ini_[i], numSMs_); 
  }
  
  for (int level_index = LEVELS-1; level_index>=finest_level_; --level_index)
  {    
    int iter_num = alignment_iterations_[level_index];    
    
    DepthMapf& depthinv_ini = depthinvs_ini_[level_index];     
    IntensityMapf& intensity_ini = intensities_ini_[level_index];     
    
    //We need gradients on the KF maps. 
    GradientMap& xGradDepthinv_ini = xGradsDepthinv_ini_[level_index];
    GradientMap& yGradDepthinv_ini = yGradsDepthinv_ini_[level_index];
    
    GradientMap& xGradIntensity_ini = xGradsIntensity_ini_[level_index];
    GradientMap& yGradIntensity_ini = yGradsIntensity_ini_[level_index];
    
    MapArr& normals_ini = normals_ini_[level_index];

    //Residuals 
    DeviceArray<float>&  res_depthinv = res_depthinvs_[level_index]; 
    DeviceArray<float>&  res_intensity = res_intensities_[level_index];    
    
    Eigen::Matrix3d cam_rot_incremental_inv;
    Eigen::Matrix3d cam_rot_incremental;
    Eigen::Vector3d cam_trans_incremental;    

    // run optim for iter_num iterations (return false when lost)
    for (int iter = 0; iter < iter_num; ++iter)
    {		
      Eigen::Vector3f init_vector = Vector3f(0,0,0);
      float3 device_current_delta_trans = device_cast<float3>(init_vector);
      float3 device_current_delta_rot = device_cast<float3>(init_vector);
      
      Eigen::Matrix3d inv_current_rotation = current_rotation.inverse();
      Eigen::Vector3d inv_current_translation = - current_rotation.inverse()*current_translation;  
      
      Eigen::Matrix3f K;
      int div = 1 << level_index; 
      
      float fx = cam_intrinsics.fx/div;
      float fy = cam_intrinsics.fy/div;
      float cx = cam_intrinsics.cx/div;
      float cy = cam_intrinsics.cy/div;
      
       K << fx,  0.f, cx,
            0.f, fy, cy,
            0.f, 0.f, 1.f;
      
      //last_iter_flag =  ((level_index == finest_level_) && (iter ==  (iter_num-1)));
      Vector3f inv_current_translation_f = K*inv_current_translation.cast<float>();
      Matrix3f inv_current_rotation_f = K*inv_current_rotation.cast<float>()*K.inverse();
      //Matrix3f current_rotation_f = current_rotation.cast<float>();
      
      float3 device_inv_current_trans = device_cast<float3>(inv_current_translation_f);
      Mat33 device_inv_current_rot = device_cast<Mat33>(inv_current_rotation_f);   
      
      
      warpInvDepthWithTrafo3D  (depthinvs_end_[level_index], warped_depthinvs_end_[level_index], depthinvs_ini_[level_index], 
                                 device_inv_current_rot, 
                                 device_inv_current_trans, 
                                 cam_intrinsics(level_index),
                                 numSMs_);  
                                 
      warpIntensityWithTrafo3DInvDepth  (intensities_end_[level_index], warped_intensities_end_[level_index], depthinvs_ini_[level_index], 
                                         device_inv_current_rot, 
                                         device_inv_current_trans, 
                                         cam_intrinsics(level_index), numSMs_);  

      DepthMapf&  warped_depthinv_end = warped_depthinvs_end_[level_index]; 
      DepthMapf&  warped_intensity_end = warped_intensities_end_[level_index]; 
      		
      computeErrorGridStride (warped_depthinv_end, depthinv_ini, res_depthinv, 19200, numSMs_);
      computeErrorGridStride (warped_intensity_end, intensity_ini, res_intensity, 19200, numSMs_); 
      
      //if ((termination_ == CHI_SQUARED)&&(!(iter == 0)))
      //if (!(iter == 0))
      //{
        //computeErrorGridStride (warped_intensities_end_[0], intensities_ini_[0], res_intensities_[0], numSMs_);			
        //computeErrorGridStride (warped_depthinvs_end_[0], depthinvs_ini_[0], res_depthinvs_[0], numSMs_);
        
        //computeChiSquare (res_intensities_[0], res_depthinvs_[0], sigma_int_ref, sigma_depthinv_ref, RGBID_SLAM::device::STUDENT, chi_square, chi_test, Ndof);
        //RMSE = sqrt(chi_square)/sqrt(Ndof);
        ////std::cout << "lvl " << level_index << " iter " << iter << ", chiSq: " << chi_square << ",Ndof: " << Ndof << ", RMSE: "<< RMSE   << std::endl;
        
        //if (!(iter == 1))
        //{        
          //if (RMSE > RMSE_prev) //undo the previous increment and end iters at curr pyr
          //{
            //current_translation = cam_rot_incremental_inv*(current_translation - cam_trans_incremental);
            //current_rotation = cam_rot_incremental_inv*current_rotation;
            ////std::cout << "Break in pyr " << level_index << " at iteration " << iter << std::endl; 
            //break;
          //}
          
          //float rel_diff = fabs(RMSE - RMSE_prev) / RMSE_prev;
          
          
          ////if (rel_diff < 0.00001) //end iters at curr pyr
          ////{
            //////std::cout << "Break in pyr " << level_index << " at iteration " << iter << std::endl; 
            ////break;
          ////}
        //}   
        
        //RMSE_prev = RMSE;
      //}
      
      
      sigma_depthinv = 0.0025f;
      bias_depthinv = 0.f;
      sigma_intensity = 5.f;
      bias_intensity = 0.f;
      nu_depthinv = 5.f;
      nu_intensity = 5.f;
      float Nsamples = 10000;
      
      //computeSigmaPdf(res_depthinv, bias_depthinv, sigma_depthinv, RGBID_SLAM::device::STUDENT); 
      
      computeNuStudent(res_depthinv, bias_depthinv, sigma_depthinv, nu_depthinv, numSMs_);
      computeNuStudent(res_intensity, bias_intensity, sigma_intensity, nu_intensity, numSMs_);
      
      nu_intensity = std::max(nu_depthinv, nu_intensity);
      

      buildSystemStudentNuGridStride (device_current_delta_trans, device_current_delta_rot,
                              depthinv_ini,  intensity_ini,
                              xGradDepthinv_ini, yGradDepthinv_ini,
                              xGradIntensity_ini, yGradIntensity_ini,		
                              warped_depthinv_end,  warped_intensity_end,	
                              RGBID_SLAM::device::STUDENT, RGBID_SLAM::device::INDEPENDENT,
                              sigma_depthinv,  sigma_intensity,
                              bias_depthinv, bias_intensity, 
                              nu_depthinv, nu_depthinv,
                              cam_intrinsics(level_index), B_SIZE,
                              gbuf_, sumbuf_, A_total.data (), b_total.data (), numSMs_);
                       
      MatrixXd A_optim(6, 6);
      A_optim = A_total.block(0,0,6,6);

      MatrixXd b_optim(6, 1);
      b_optim = b_total.block(0,0,6,1);  

      MatrixXd result(6, 1);
      result = A_optim.llt().solve(b_optim);

      //Ilumination correction variables affect the optimisation result when used, but apart from that we dont do anything with them.
      //This part remains equal even if we used ilumination change parameters
      Eigen::Vector3d res_trans = result.block(0,0,3,1).cast<double>();	
      Eigen::Vector3d res_rot = result.block(3,0,3,1).cast<double>();	

      //If output is B_theta^A and r_B^A
      Matrix3ft rot_ft = expMapRot(res_rot.cast<float_type>());
      cam_rot_incremental_inv = rot_ft.cast<double>();
      cam_rot_incremental = cam_rot_incremental_inv.inverse();
      cam_trans_incremental = -cam_rot_incremental*res_trans;

      //Transform updates are applied by premultiplying. Seems counter-intuitive but it is the correct way,
      //since at each iter we warp curr frame towards prev frame.
      current_translation = cam_rot_incremental * current_translation + cam_trans_incremental;
      current_rotation = cam_rot_incremental * current_rotation;
    } 
  } 
  
  MatrixXd A_final(6, 6);
  A_final = A_total.block(0,0,6,6);
  
  rotation_ini2end = current_rotation;
  translation_ini2end = current_translation; 
  
  //Eigen::Matrix<double,6,6> pseudo_adj = Eigen::Matrix<double,6,6>::Zero();
  //pseudo_adj.block<3,3>(0,0) = -rotation_ini2end.cast<double>();
  //pseudo_adj.block<3,3>(3,3) = -rotation_ini2end.cast<double>();
  
  //covariance_ini2end = pseudo_adj*A_final.inverse()*pseudo_adj.transpose(); 
  covariance_ini2end = A_final.inverse(); 
  
  
  //rotation_ini2end = current_rotation*rotation_ini2end;
  //translation_ini2end = current_rotation*translation_ini2end + current_translation;  
  
  return true;
}


