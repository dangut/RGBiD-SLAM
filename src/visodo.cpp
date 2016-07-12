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
#include "visodo.h"
#include "util_funcs.h"
#include "keyframe_manager.h"
#include "pose_graph_manager.h"

//#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#ifdef HAVE_OPENCV
  #include <opencv2/opencv.hpp>
  //~ #include <opencv2/gpu/gpu.hpp>
  //~ #include <pcl/gpu/utils/timers_opencv.hpp>
#endif

using namespace RGBID_SLAM::device;

using RGBID_SLAM::device::device_cast;


                       
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
RGBID_SLAM::VisodoTracker::VisodoTracker (int optim_dim, int Mestimator, int motion_model, int sigma_estimator, int weighting, int warping, int max_odoKF_count, int finest_level, int termination, float visratio_odo, int image_filtering, float visratio_integr, int max_integrKF_count,  int Nsamples, int rows, int cols) : 
  rows_(rows), cols_(cols),  global_time_(0), lost_ (false), optim_dim_(optim_dim), Mestimator_(Mestimator), motion_model_(motion_model), sigma_estimator_(sigma_estimator), weighting_(weighting), warping_(warping), max_odoKF_count_(max_odoKF_count), finest_level_(finest_level), termination_(termination), visibility_ratio_odo_threshold_(visratio_odo), image_filtering_(image_filtering), visibility_ratio_integr_threshold_(visratio_integr), max_integrKF_count_(max_integrKF_count), Nsamples_(Nsamples)
{   
  setRGBIntrinsics (FOCAL_LENGTH, FOCAL_LENGTH, CENTER_X, CENTER_Y);
  setDepthIntrinsics (FOCAL_LENGTH, FOCAL_LENGTH_DEPTH, CENTER_X, CENTER_Y);
  //setDepthIntrinsics (FOCAL_LENGTH_DEPTH, FOCAL_LENGTH_DEPTH, CENTER_X, CENTER_Y);
  
  init_Rcam_ = Matrix3ft::Identity ();// * Eigen::AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = Vector3ft (0.f, 0.f, 0.f);
  
  dRc_ = Matrix3f::Identity();
  t_dc_ = Vector3f(0.f, 0.f, 0.f);
  //t_dc_ = Vector3f(0.0254f, 0.f, 0.f);
  
  custom_registration_ = 0;
  
  const int iters[] = {10, 5, 3, 0, 0, 0};
  
  num_prev_integr_ = 50;
  
  std::copy (iters, iters + LEVELS, visodo_iterations_);
  real_time_flag_ = 0;
  
  allocateBuffers (rows, cols);

  rmats_.reserve (30000);
  tvecs_.reserve (30000);
  vis_odo_times_.reserve(30000);  
  
  timestamps_.reserve(30000);
  odo_rmats_.reserve(30000);
  odo_tvecs_.reserve(30000);
  odo_covmats_.reserve(30000);
  
  rmatsKF_.reserve(200);
  tvecsKF_.reserve(200);
  
  scene_view_.resize(cols_*rows_);
  intensity_view_.resize(cols_*rows_,0.f);
  depthinv_view_.resize(cols_*rows_);
  
  newKF_ = false;
  scene_view_has_changed_ = false;
  camera_pose_has_changed_ = false;
  
  factor_depth_ = 1.f;

  reset ();  
}

void RGBID_SLAM::VisodoTracker::loadCalibration(std::string const &calib_file)
{
  std::ifstream filestream(calib_file.c_str());
  
  if (!filestream.is_open())
  {
    std::cout << "Could not open configuration file " << calib_file << std::endl;
    return;
  }
    
  Settings settings(filestream);
  Section calibration;
  
  if (settings.getSection("CALIBRATION",calibration) || settings.getSection("RGB_CALIBRATION",calibration))
  {
    std::cout << "RGB CALIBRATION" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    Entry entry;
    
    if (calibration.getEntry("fx",entry))
    {
      //std::string fx_str = entry.getValue();
      std::stringstream fx_ss(entry.getValue());
      
      fx_ss >> fx_;
      fxd_ = fx_;
      
      //fx_ = atof(fx_str.c_str());
      
      std::cout << "  FOCAL X: " << fx_ << std::endl;    
    }
    
    if (calibration.getEntry("fy",entry))
    {
      std::string fy_str = entry.getValue();
      
      fy_ = atof(fy_str.c_str());
      fyd_ = fy_;
      
      std::cout << "  FOCAL Y: " << fy_ << std::endl;    
    }
    
    if (calibration.getEntry("cx",entry))
    {
      std::string cx_str = entry.getValue();
      
      cx_ = atof(cx_str.c_str());
      cxd_ = cx_;
      
      std::cout << "  CENTER X: " << cx_ << std::endl;    
    }
    
    if (calibration.getEntry("cy",entry))
    {
      std::string cy_str = entry.getValue();
      
      cy_ = atof(cy_str.c_str());
      cyd_ = cy_;
      
      std::cout << "  CENTER Y: " << cy_ << std::endl;    
    }
    
    if (calibration.getEntry("kd",entry))
    {
      std::stringstream kd_ss(entry.getValue());
      
      kd_ss >> k1_ >> k2_ >> k3_ >> k4_ >> k5_ ;
      
      std::cout << "  DISTORTION COEFS: " << k1_ << " " << k2_ << " " << k3_ << " " << k4_ << " " << k5_ << " " << std::endl;    
    }
    
    
    if (calibration.getEntry("factor_depth",entry))
    {
      std::string factor_depth_str = entry.getValue();
      
      factor_depth_ = atof(factor_depth_str.c_str());
      
      std::cout << "  factor_depth: " << factor_depth_ << std::endl;    
    }
  }
      
  if (settings.getSection("DEPTH_CALIBRATION",calibration))
  {
    std::cout << "DEPTH CALIBRATION" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    Entry entry;
    
    if (calibration.getEntry("custom_registration",entry))
    {
      //std::string fx_str = entry.getValue();
      std::stringstream cr_ss(entry.getValue());
      
      cr_ss >> custom_registration_;
      
      //fx_ = atof(fx_str.c_str());
      
      std::cout << "  custom_registration: " << (custom_registration_>0) << std::endl;    
    }
    
    if (calibration.getEntry("fx",entry))
    {
      //std::string fx_str = entry.getValue();
      std::stringstream fx_ss(entry.getValue());
      
      fx_ss >> fxd_;
      
      //fx_ = atof(fx_str.c_str());
      
      std::cout << "  FOCAL X: " << fxd_ << std::endl;    
    }
    
    if (calibration.getEntry("fy",entry))
    {
      std::string fy_str = entry.getValue();
      
      fyd_ = atof(fy_str.c_str());
      
      std::cout << "  FOCAL Y: " << fyd_ << std::endl;    
    }
    
    if (calibration.getEntry("cx",entry))
    {
      std::string cx_str = entry.getValue();
      
      cxd_ = atof(cx_str.c_str());
      
      std::cout << "  CENTER X: " << cxd_ << std::endl;    
    }
    
    if (calibration.getEntry("cy",entry))
    {
      std::string cy_str = entry.getValue();
      
      cyd_ = atof(cy_str.c_str());
      
      std::cout << "  CENTER Y: " << cyd_ << std::endl;    
    }
    
    if (calibration.getEntry("kd",entry))
    {
      std::stringstream kd_ss(entry.getValue());
      
      kd_ss >> k1d_ >> k2d_ >> k3d_ >> k4d_ >> k5d_ ;
      
      std::cout << "  DISTORTION COEFS: " << k1d_ << " " << k2d_ << " " << k3d_ << " " << k4d_ << " " << k5d_ << " " << std::endl;    
    }
    
    if (calibration.getEntry("c0",entry))
    {
      std::stringstream c0_ss(entry.getValue());
      
      c0_ss >> c0_ ;
      
      std::cout << "  c0: " << c0_ << " " << std::endl;    
    }
    
    if (calibration.getEntry("c1",entry))
    {
      std::stringstream c1_ss(entry.getValue());
      
      c1_ss >> c1_ ;
      
      std::cout << "  c1: " << c1_ << " " << std::endl;    
    }
    
    if (calibration.getEntry("alpha0",entry))
    {
      std::stringstream alpha0_ss(entry.getValue());
      
      alpha0_ss >> alpha0_ ;
      
      std::cout << "  alpha0: " << alpha0_ << " " << std::endl;    
    }
    
    if (calibration.getEntry("alpha1",entry))
    {
      std::stringstream alpha1_ss(entry.getValue());
      
      alpha1_ss >> alpha1_ ;
      
      std::cout << "  alpha1: " << alpha1_ << " " << std::endl;    
    }
    
    if (calibration.getEntry("alpha2",entry))
    {
      std::stringstream alpha2_ss(entry.getValue());
      
      alpha2_ss >> alpha2_ ;
      
      std::cout << "  alpha2: " << alpha2_ << " " << std::endl;    
    }
    
    if (calibration.getEntry("spdist",entry))
    {
      std::stringstream kspd_ss(entry.getValue());
      
      kspd_ss >> kspd1_ >> kspd2_ >> kspd3_ >> kspd4_ >> kspd5_ >> kspd6_ >> kspd7_ >> kspd8_ ;
      
      std::cout << "  DEPTH DIST COEFS: " << kspd1_ << " " << kspd2_ << " " << kspd3_ << " " << kspd4_ << " " << kspd5_ << " " << kspd6_ << " " << kspd7_ << " " << kspd8_ << " " << std::endl;    
    }
        
    //TODO: setDepthIntrinsics (fx_, fy_, cx_, cy_);    
  }
  
  if (settings.getSection("STEREO_DEPTH2RGB",calibration))
  {
    std::cout << "STEREO DEPTH-to-RGB CALIBRATION" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    Entry entry;
    
    if (calibration.getEntry("dRc",entry))
    {
      std::stringstream dRc_ss(entry.getValue());
      dRc_ss >> dRc_(0,0) >> dRc_(0,1) >> dRc_(0,2)
             >> dRc_(1,0) >> dRc_(1,1) >> dRc_(1,2) 
             >> dRc_(2,0) >> dRc_(2,1) >> dRc_(2,2) ;
              
      std::cout << "  STEREO dRc: " << std::endl
                << dRc_ << std::endl;
    }
    
    if (calibration.getEntry("t_dc",entry))
    {
      std::stringstream dRc_ss(entry.getValue());
      dRc_ss >> t_dc_(0) >> t_dc_(1) >> t_dc_(2);
              
      std::cout << "  STEREO t_dc: " <<  t_dc_.transpose() << std::endl;
    }
  }
  
}
  
void RGBID_SLAM::VisodoTracker::loadSettings (Settings &settings)
{
  Section visodo_section;
  
  if (settings.getSection("VISODO",visodo_section))
  {
    std::cout << "VISODO" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    Entry entry;
    
    if (visodo_section.getEntry("M_ESTIMATOR",entry))
    {
      std::string Mestimator_str = entry.getValue();
      
      if (Mestimator_str.compare("Student") == 0)   //enum int type defined in internal.h
		    Mestimator_ = RGBID_SLAM::device::STUDENT;
      if (Mestimator_str.compare("LeastSquares") == 0)   //enum int type defined in internal.h
		    Mestimator_ = RGBID_SLAM::device::LSQ;
      if (Mestimator_str.compare("Tukey") == 0)   //enum int type defined in internal.h
        Mestimator_ = RGBID_SLAM::device::TUKEY;
      if (Mestimator_str.compare("Huber") == 0)   //enum int type defined in internal.h
        Mestimator_ = RGBID_SLAM::device::HUBER;
        
      std::cout << "  M_ESTIMATOR: " << Mestimator_str <<  std::endl;
    
    }
    
    if (visodo_section.getEntry("MOTION_MODEL",entry))
    {
      std::string mm_str = entry.getValue();
      
      if (mm_str.compare("none") == 0)   //enum int type defined in internal.h
		    motion_model_ = RGBID_SLAM::device::NO_MM;
      if (mm_str.compare("constVelocity") == 0)   //enum int type defined in internal.h
		    motion_model_ = RGBID_SLAM::device::CONSTANT_VELOCITY;   
        
      std::cout << "  MOTION_MODEL: " << mm_str << std::endl;    
    }
    
    if (visodo_section.getEntry("WARP_ORDER",entry))
    {
      std::string warp_str = entry.getValue();
      
      if (warp_str.compare("warpFirst") == 0)   //enum int type defined in internal.h
		    warping_ = RGBID_SLAM::device::WARP_FIRST;
      if (warp_str.compare("pyrFirst") == 0)   //enum int type defined in internal.h
		    warping_ = RGBID_SLAM::device::PYR_FIRST;   
        
      std::cout << "  WARP_ORDER: " << warp_str << std::endl;    
    }
    
    if (visodo_section.getEntry("IMAGE_FILTERING",entry))
    {
      std::string filtering_str = entry.getValue();
      
      if (filtering_str.compare("none") == 0)   //enum int type defined in internal.h
		    image_filtering_ = RGBID_SLAM::device::NO_FILTERS;
      if (filtering_str.compare("gradients") == 0)   //enum int type defined in internal.h
		    image_filtering_ = RGBID_SLAM::device::FILTER_GRADS;   
        
      std::cout << "  IMAGE_FILTERING: " << filtering_str << std::endl;    
    }
    
    
    if (visodo_section.getEntry("SIGMA_ESTIMATOR",entry))
    {
      std::string sigma_estimator_str = entry.getValue();
      
      if (sigma_estimator_str.compare("sigmaMAD") == 0)   //enum int type defined in internal.h
		    sigma_estimator_ = RGBID_SLAM::device::SIGMA_MAD;
      if (sigma_estimator_str.compare("sigmaML") == 0)   //enum int type defined in internal.h
		    sigma_estimator_ = RGBID_SLAM::device::SIGMA_PDF;
      if (sigma_estimator_str.compare("sigmaConst") == 0)   //enum int type defined in internal.h
        sigma_estimator_ = RGBID_SLAM::device::SIGMA_CONS;
        
      std::cout << "  SIGMA_ESTIMATOR: " << sigma_estimator_str << std::endl;
    
    }
    
    if (visodo_section.getEntry("INTEGRATION_VISRATIO_THRESHOLD",entry))
    {
      std::string visratio_integr_str = entry.getValue();
      
      visibility_ratio_integr_threshold_ = atof(visratio_integr_str.c_str());
      
      std::cout << "  INTEGRATION_VISRATIO_THRESHOLD: " << visibility_ratio_integr_threshold_ << std::endl;    
    }
    
    if (visodo_section.getEntry("ODOMETRY_VISRATIO_THRESHOLD",entry))
    {
      std::string visratio_odo_str = entry.getValue();
      
      visibility_ratio_odo_threshold_ = atof(visratio_odo_str.c_str());
      
      std::cout << "  ODOMETRY_VISRATIO_THRESHOLD: " << visibility_ratio_odo_threshold_ << std::endl;
    
    }
    
    if (visodo_section.getEntry("FINEST_PYR_LEVEL",entry))
    {
      std::string finest_level_str = entry.getValue();
      
      finest_level_ = atof(finest_level_str.c_str());
      
      std::cout << "  FINEST_PYR_LEVEL: " << finest_level_ << std::endl;
    }
    
    
    std::cout << std::endl;
    std::cout << std::endl;
    
  }
}

void
RGBID_SLAM::VisodoTracker::start()
{
  boost::unique_lock<boost::mutex> lock(created_aux_mutex_);
  
  visodo_thread_.reset(new boost::thread(boost::ref(*this)));
  
  created_cond_.wait (lock);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
RGBID_SLAM::VisodoTracker::setRGBIntrinsics (float fx, float fy, float cx, float cy, float k1, float k2, float k3, float k4, float k5)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = (cx == -1) ? cols_/2-0.5f : cx;
  cy_ = (cy == -1) ? rows_/2-0.5f : cy; 
   
  k1_ = k1;
  k2_ = k2;
  k3_ = k3;
  k4_ = k4;
  k5_ = k5;
}

void
RGBID_SLAM::VisodoTracker::setDepthIntrinsics (float fxd, float fyd, float cxd, float cyd, float k1d, float k2d, float k3d, float k4d, float k5d,
                                                float c0, float c1, float alpha0, float alpha1, float alpha2, 
                                                float kspd1, float kspd2, float kspd3, float kspd4, 
                                                float kspd5, float kspd6, float kspd7, float kspd8)
{
  fxd_ = fxd;
  fyd_ = fyd;
  cxd_ = (cxd == -1) ? cols_/2-0.5f : cxd;
  cyd_ = (cyd == -1) ? rows_/2-0.5f : cyd;  
  
  k1d_ = k1d;
  k2d_ = k2d;
  k3d_ = k3d;
  k4d_ = k4d;
  k5d_ = k5d;
  
  c0_ = c0;
  c1_ = c1;
  alpha0_ = alpha0;
  alpha1_ = alpha1;
  alpha2_ = alpha2;
  
  kspd1_ = kspd1;
  kspd2_ = kspd2;
  kspd3_ = kspd3;
  kspd4_ = kspd4;
  kspd5_ = kspd5;
  kspd6_ = kspd6;
  kspd7_ = kspd7;
  kspd8_ = kspd8;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
RGBID_SLAM::VisodoTracker::setSharedCameraPose (const Eigen::Affine3d& pose)
{
  boost::mutex::scoped_lock lock(mutex_shared_camera_pose_);  
  shared_camera_pose_ = pose;
  camera_pose_has_changed_ = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
RGBID_SLAM::VisodoTracker::cols ()
{
  return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
RGBID_SLAM::VisodoTracker::rows ()
{
  return (rows_);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
RGBID_SLAM::VisodoTracker::reset ()
{
  cout << "reseting..." << std::endl;
  
  global_time_ = 0;
  rmats_.clear ();
  tvecs_.clear ();
  vis_odo_times_.clear();
  timestamps_.clear();

  rmats_.push_back (init_Rcam_);
  tvecs_.push_back (init_tcam_);
  vis_odo_times_.push_back(0.f);
 
  // reset estimated pose
  last_estimated_rotation_= Matrix3ft::Identity ();
  last_estimated_translation_= Vector3ft (0.f, 0.f, 0.f);  
  
  velocity_= Vector3ft (0.f, 0.f, 0.f);
  omega_= Vector3ft (0.f, 0.f, 0.f);
  
  initialiseBufferPointers();
  
  cout << "reseting2..." << std::endl;
    
  lost_=false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
RGBID_SLAM::VisodoTracker::getImage (std::vector<PixelRGB>& scene_view, std::vector<float>& intensity_view, std::vector<float>& depthinv_view ) 
{
  Eigen::Vector3f light_source_pose = last_integrKF_global_translation_.cast<float>();

  LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);

  View scene_view_dev;
  scene_view_dev.create (rows_, cols_ );
  
  generateImageRGB (vertices_integrKF_, normals_integrKF_, colors_integrKF_, light, scene_view_dev);
  //generateImage (vertices_odoKF_[0], normals_odoKF_[0],  light, scene_view_dev);
  //generateImage (vertices_integrKF_, normals_integrKF_, light, scene_view_dev);
  
  RGBID_SLAM::device::sync ();
  
  scene_view_dev.download(scene_view, cols_);
  intensities_curr_[0].download(intensity_view, cols_);
  depthinv_integrKF_.download(depthinv_view, cols_);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
RGBID_SLAM::VisodoTracker::allocateBuffers (int rows, int cols)
{    

  std::cout << "Memory usage BEFORE allocating buffers" << std::endl;
  showGPUMemoryUsage();
  intensity_debug_.create(rows,cols);
  
  depthinvs_curr_.resize (LEVELS);
  intensities_curr_.resize(LEVELS);
  
  depthinvs_odoKF_.resize (LEVELS);
  intensities_odoKF_.resize(LEVELS);
  
  depthinvs_odoKF_filtered_.resize (LEVELS);
  intensities_odoKF_filtered_.resize(LEVELS);

  xGradsInt_odoKF_.resize(LEVELS);
  yGradsInt_odoKF_.resize(LEVELS);
  xGradsDepthinv_odoKF_.resize(LEVELS);
  yGradsDepthinv_odoKF_.resize(LEVELS);
  vertices_odoKF_.resize(LEVELS);
  normals_odoKF_.resize(LEVELS);
  
  xGradsInt_odoKF_covOnly_.resize(LEVELS);
  yGradsInt_odoKF_covOnly_.resize(LEVELS);
  xGradsDepthinv_odoKF_covOnly_.resize(LEVELS);
  yGradsDepthinv_odoKF_covOnly_.resize(LEVELS);

  warped_depthinvs_curr_.resize(LEVELS);
  warped_intensities_curr_.resize(LEVELS);  

  res_intensities_.resize(LEVELS);
  res_depthinvs_.resize(LEVELS);
  
  depthinv_distorted_.create(rows,cols);
  intensity_distorted_.create(rows,cols);
  depthinv_corr_distorted_.create(rows,cols);
  depthinv_preregister_.create(rows,cols);
  depthinv_register_trans_.create(3*rows,3*cols);
  depthinv_register_trans_as_int_.create(3*rows,3*cols);
  
  projected_transformed_points_.create(rows*2,cols);
  depthinv_warped_in_curr_.create(rows,cols);
  warped_weight_curr_.create(rows,cols);
  warped_depthinv_integr_curr_.create(rows,cols);
  depthinv_integrKF_.create(rows,cols);
  weight_integrKF_.create(rows,cols);
  overlap_mask_integrKF_.create(rows,cols);
  depthinv_integrKF_raw_.create(rows,cols);
  vertices_integrKF_.create(3*rows,cols);
  normals_integrKF_.create(3*rows,cols);
  xGradsDepthinv_integrKF_.create(rows,cols);
  yGradsDepthinv_integrKF_.create(rows,cols);
  colors_integrKF_.create(rows,cols);
  
  r_curr_.create(rows,cols);
  b_curr_.create(rows,cols);
  g_curr_.create(rows,cols);
  
  warped_r_curr_.create(rows,cols);
  warped_b_curr_.create(rows,cols);
  warped_g_curr_.create(rows,cols);
  
  warped_int_RGB_.create(rows,cols);  
  
  rect_corner_pos_.create(4*rows,cols);
  
  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    intensities_curr_[i].create (pyr_rows, pyr_cols);
    depthinvs_curr_[i].create (pyr_rows, pyr_cols);
    
    intensities_odoKF_[i].create (pyr_rows, pyr_cols);
    depthinvs_odoKF_[i].create (pyr_rows, pyr_cols);
    
    intensities_odoKF_filtered_[i].create (pyr_rows, pyr_cols);
    depthinvs_odoKF_filtered_[i].create (pyr_rows, pyr_cols);  

    xGradsInt_odoKF_[i].create (pyr_rows, pyr_cols);
    yGradsInt_odoKF_[i].create (pyr_rows, pyr_cols);
    xGradsDepthinv_odoKF_[i].create (pyr_rows, pyr_cols);
    yGradsDepthinv_odoKF_[i].create (pyr_rows, pyr_cols); 
    vertices_odoKF_[i].create (3*pyr_rows, pyr_cols); 
    normals_odoKF_[i].create (3*pyr_rows, pyr_cols); 
    
    xGradsInt_odoKF_covOnly_[i].create (pyr_rows, pyr_cols);
    yGradsInt_odoKF_covOnly_[i].create (pyr_rows, pyr_cols);
    xGradsDepthinv_odoKF_covOnly_[i].create (pyr_rows, pyr_cols);
    yGradsDepthinv_odoKF_covOnly_[i].create (pyr_rows, pyr_cols); 
    
    warped_depthinvs_curr_[i].create(pyr_rows, pyr_cols);
    warped_intensities_curr_[i].create(pyr_rows, pyr_cols);
    
    res_intensities_[i].create(pyr_rows * pyr_cols);
    res_depthinvs_[i].create(pyr_rows * pyr_cols);	
  }  
  
  // see estimate tranform for the magic numbers
  gbuf_.create (27, 20*60);
  sumbuf_.create (27);
  
  std::cout << "Memory usage AFTER allocating buffers" << std::endl;
  showGPUMemoryUsage();
}

inline void 
RGBID_SLAM::VisodoTracker::convertTransforms (Matrix3ft& rotation_in_1, Matrix3ft& rotation_in_2, Vector3ft& translation_in_1, Vector3ft& translation_in_2, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out_1, float3& translation_out_2)
{
  Matrix3f rotation_in_1f = rotation_in_1.cast<float>();
  Matrix3f rotation_in_2f = rotation_in_2.cast<float>();
  Vector3f translation_in_1f = translation_in_1.cast<float>();
  Vector3f translation_in_2f = translation_in_2.cast<float>();
  
  rotation_out_1 = device_cast<Mat33> (rotation_in_1f);
  rotation_out_2 = device_cast<Mat33> (rotation_in_2f);
  translation_out_1 = device_cast<float3>(translation_in_1f);
  translation_out_2 = device_cast<float3>(translation_in_2f);
}

inline void 
RGBID_SLAM::VisodoTracker::convertTransforms (Matrix3ft& rotation_in_1, Matrix3ft& rotation_in_2, Vector3ft& translation_in, Mat33& rotation_out_1, Mat33& rotation_out_2, float3& translation_out)
{
  Matrix3f rotation_in_1f = rotation_in_1.cast<float>();
  Matrix3f rotation_in_2f = rotation_in_2.cast<float>();
  Vector3f translation_in_f = translation_in.cast<float>();
  
  rotation_out_1 = device_cast<Mat33> (rotation_in_1f);
  rotation_out_2 = device_cast<Mat33> (rotation_in_2f);
  translation_out = device_cast<float3>(translation_in_f);
}

inline void 
RGBID_SLAM::VisodoTracker::convertTransforms (Matrix3ft& rotation_in, Vector3ft& translation_in, Mat33& rotation_out, float3& translation_out)
{
  Matrix3f rotation_in_f = rotation_in.cast<float>();
  Vector3f translation_in_f = translation_in.cast<float>();
  
  rotation_out = device_cast<Mat33> (rotation_in_f);  
  translation_out = device_cast<float3>(translation_in_f);
}

void
RGBID_SLAM::VisodoTracker::downloadAndSaveDepth(const DeviceArray2D<float>& dev_image, int pyrlevel, int iterat){

	std::vector<float> data;
	std::vector<unsigned short> data_im;
	cv::Mat cv_image;
	int cols = dev_image.cols();
	int rows = dev_image.rows();
	char im_file[128];
	data.resize(cols*rows);
	data_im.resize(cols*rows);
	int elem_step = 1;
	dev_image.download(data, elem_step);
	float aux_data;	

	for (int i=0; i < data.size(); i++){
	   aux_data = 1.f / ((float) data[i]); 
	   if (isnan(aux_data))
	      data_im[i] = 0;
	   else
	      data_im[i] = (unsigned short) (6553.5*(aux_data ));  
	}
	
  cv_image.create(rows, data_im.size() / rows, CV_16UC1);
	memcpy( cv_image.data, &data_im[0], data_im.size()*sizeof(unsigned short));
	sprintf(im_file, "depth/globalTime%05d_level%03d_iter%03d.ppm", global_time_, pyrlevel, iterat);

 	cv::imwrite(im_file, cv_image);
}


inline void 
RGBID_SLAM::VisodoTracker::prepareImages (const DepthMap& depth_raw, const View& colors_raw, const Intr& cam_intrinsics)
{  
  computeIntensity (colors_raw, intensities_curr_[0]);
  decomposeRGBInChannels(colors_raw, r_curr_, g_curr_, b_curr_ );
  convertDepth2InvDepth(depth_raw, depthinvs_curr_[0],factor_depth_);
  
  
  for (int i = 1; i < LEVELS; ++i)
  {
    pyrDownIntensity (intensities_curr_[i-1], intensities_curr_[i]);
    pyrDownDepth (depthinvs_curr_[i-1], depthinvs_curr_[i]);
  }   
}

inline void 
RGBID_SLAM::VisodoTracker::prepareImagesCustomCalibration (const DepthMap& depth_raw, const View& colors_raw, 
                                                              const Intr& rgb_intrinsics, const Intr& depth_intrinsics, const DepthDist& depth_spdist)
{  
  float timeRegister = 0.f;
  
  computeIntensity (colors_raw, intensity_distorted_);
  decomposeRGBInChannels(colors_raw, r_curr_, g_curr_, b_curr_ );
  convertDepth2InvDepth(depth_raw, depthinv_distorted_,factor_depth_);
  
  //registration in GPU (optional, when customised RGB-D calibration is available)
  timeRegister +=undistortIntensity(intensity_distorted_, intensities_curr_[0], rgb_intrinsics);
  timeRegister +=undistortDepthInv(depthinv_distorted_, depthinv_corr_distorted_, depthinv_preregister_, depth_intrinsics, depth_spdist);
  
  float3 t_dc_proj_dev;
  Mat33 cRd_proj_dev, dRc_proj_dev;  
  
  Eigen::Matrix3f Kc = getCalibMatrix(0); //TODO
  Eigen::Matrix3f Kd = getCalibMatrixDepth(0);  //TODO
  
  Matrix3f dRc_proj_f = Kd*dRc_*Kc.inverse();
  Vector3f t_dc_proj_f = Kd*t_dc_;
  
  //dRc_proj_f = Matrix3f::Identity();
  //t_dc_proj_f << 0.f, 0.f, 0.f;
  
  Matrix3f cRd_proj_f = dRc_proj_f.inverse();
  
  std::cout << t_dc_ << std::endl;
  
  t_dc_proj_dev = device_cast<float3>(t_dc_proj_f);
  cRd_proj_dev = device_cast<Mat33>(cRd_proj_f); 
  dRc_proj_dev = device_cast<Mat33>(dRc_proj_f);
  
  
  
  timeRegister += registerDepthinv(depthinv_preregister_, depthinv_register_trans_, depthinv_register_trans_as_int_, depthinvs_curr_[0], dRc_proj_dev, t_dc_proj_dev, cRd_proj_dev);
  //timeRegister += registerDepthinv(depthinv_preregister_, depthinv_preregister_, depthinv_register_trans_as_int_, depthinvs_curr_[0], dRc_proj_dev, t_dc_proj_dev, cRd_proj_dev);
  //downloadAndSaveDepth( depthinvs_curr_[0]);
  //depthinv_preregister_.copyTo(depthinvs_curr_[0]);
  std::cout << "Time registration: " << timeRegister << std::endl;
  //////////////
  
  for (int i = 1; i < LEVELS; ++i)
  {
    pyrDownIntensity (intensities_curr_[i-1], intensities_curr_[i]);
    pyrDownDepth (depthinvs_curr_[i-1], depthinvs_curr_[i]);
  }   
}


inline void
RGBID_SLAM::VisodoTracker::saveCurrentImagesAsOdoKeyframes  (const Intr cam_intrinsics)
{
  float timeFilters = 0.f;
  float timeGradients = 0.f;
  float timeWarping = 0.f;
  
  float sigma_int_ref, sigma_depthinv_ref;
  sigma_int_ref = 3.f;
  sigma_depthinv_ref = 0.0025f;		
  
  for (int i = 0; i < LEVELS; ++i)
  {   
    copyImages (depthinvs_curr_[i], intensities_curr_[i], depthinvs_odoKF_[i], intensities_odoKF_[i]);
  } 
  
  timeFilters += bilateralFilter(depthinvs_odoKF_[0], depthinvs_odoKF_filtered_[0], 2.f*sigma_depthinv_ref);
  timeFilters += bilateralFilter(intensities_odoKF_[0], intensities_odoKF_filtered_[0], sigma_int_ref);
  
  //copyImages (depthinvs_odoKF_[0], intensities_odoKF_[0], depthinvs_odoKF_filtered_[0], intensities_odoKF_filtered_[0]);
  
  timeGradients += computeGradientIntensity(intensities_odoKF_filtered_[0], xGradsInt_odoKF_covOnly_[0], yGradsInt_odoKF_covOnly_[0]);
  timeGradients += computeGradientDepth(depthinvs_odoKF_filtered_[0], xGradsDepthinv_odoKF_covOnly_[0], yGradsDepthinv_odoKF_covOnly_[0]);
  
  for (int i = 1; i < LEVELS; ++i)
  {     
  timeWarping+=pyrDownDepth (depthinvs_odoKF_filtered_[i-1], depthinvs_odoKF_filtered_[i]);
  timeWarping+=pyrDownIntensity (intensities_odoKF_filtered_[i-1], intensities_odoKF_filtered_[i]);
  
  timeGradients += computeGradientIntensity(intensities_odoKF_filtered_[i], xGradsInt_odoKF_covOnly_[i], yGradsInt_odoKF_covOnly_[i]);
  timeGradients += computeGradientDepth(depthinvs_odoKF_filtered_[i], xGradsDepthinv_odoKF_covOnly_[i], yGradsDepthinv_odoKF_covOnly_[i]);
  }     
  
  if (image_filtering_ == FILTER_GRADS)  
  {
    for (int i = 0; i < LEVELS; ++i)
    {     
      copyImages (xGradsInt_odoKF_covOnly_[i], yGradsInt_odoKF_covOnly_[i], xGradsInt_odoKF_[i], yGradsInt_odoKF_[i]);
      copyImages (xGradsDepthinv_odoKF_covOnly_[i], yGradsDepthinv_odoKF_covOnly_[i], xGradsDepthinv_odoKF_[i], yGradsDepthinv_odoKF_[i]);
    }  
  }
  else
  {
    for (int i = 0; i < LEVELS; ++i)
    {     
      timeGradients += computeGradientIntensity(intensities_odoKF_[i], xGradsInt_odoKF_[i], yGradsInt_odoKF_[i]);
      timeGradients += computeGradientDepth(depthinvs_odoKF_[i], xGradsDepthinv_odoKF_[i], yGradsDepthinv_odoKF_[i]); 
      //createVMap( cam_intrinsics(i), depthinvs_odoKF_[i], vertices_odoKF_[i]);
      //createNMap( vertices_odoKF_[i], normals_odoKF_[i]);
      //createNMapGradients (cam_intrinsics(i), depthinvs_odoKF_[i], xGradsDepthinv_odoKF_[i], yGradsDepthinv_odoKF_[i], normals_odoKF_[i]);
    }
  }    
}

inline void
RGBID_SLAM::VisodoTracker::saveCurrentImagesAsIntegrationKeyframes  (const Intr cam_intrinsics, const View& colors)
{
  copyImage (depthinvs_curr_[0], depthinv_integrKF_);
  copyImage (depthinvs_curr_[0], depthinv_integrKF_raw_);
  colors.copyTo(colors_integrKF_);
   
  initialiseWeightKeyframe(depthinvs_curr_[0], weight_integrKF_);
  
  createVMap( cam_intrinsics(0), depthinv_integrKF_, vertices_integrKF_);
  //createNMap( vertices_integrKF_, normals_integrKF_);  
  computeGradientDepth(depthinv_integrKF_, xGradsDepthinv_integrKF_, yGradsDepthinv_integrKF_);    
  createNMapGradients( cam_intrinsics(0), depthinv_integrKF_, xGradsDepthinv_integrKF_, yGradsDepthinv_integrKF_, normals_integrKF_);     	
}

inline void
RGBID_SLAM::VisodoTracker::saveIntegrationKeyframesAsOdoKeyframes  (const Intr cam_intrinsics)
{
    
  copyImage (depthinv_integrKF_, depthinvs_odoKF_[0]);    
    
  for (int i = 1; i < LEVELS; ++i)
  {     
    pyrDownDepth (depthinvs_odoKF_[i-1], depthinvs_odoKF_[i]);    
  }
  
  for (int i = 0; i < LEVELS; ++i)
  {     
    computeGradientDepth(depthinvs_odoKF_[i], xGradsDepthinv_odoKF_[i], yGradsDepthinv_odoKF_[i]); 
    createVMap( cam_intrinsics(i), depthinvs_odoKF_[i], vertices_odoKF_[i]);
    createNMap( vertices_odoKF_[i], normals_odoKF_[i]);
  }
    	
}


inline void 
RGBID_SLAM::VisodoTracker::initialiseBufferPointers()
{  
  pop_depthinv_buffer_ptr = &depthinv_buffer2;
  push_depthinv_buffer_ptr = &depthinv_buffer1;  
}


inline void 
RGBID_SLAM::VisodoTracker::swapBufferPointers()
{
  if ((pop_depthinv_buffer_ptr == &depthinv_buffer1) && (push_depthinv_buffer_ptr == &depthinv_buffer2) )
  {
    pop_depthinv_buffer_ptr = &depthinv_buffer2;
    push_depthinv_buffer_ptr = &depthinv_buffer1;
  } 
  else if ((pop_depthinv_buffer_ptr == &depthinv_buffer2) && (push_depthinv_buffer_ptr == &depthinv_buffer1) )
  {
    pop_depthinv_buffer_ptr = &depthinv_buffer1;
    push_depthinv_buffer_ptr = &depthinv_buffer2;
  } 
  else
  {
    //TODO error 
  }
}
 

inline bool 
RGBID_SLAM::VisodoTracker::estimateVisualOdometry  (const Intr cam_intrinsics, Matrix3ft& resulting_rotation , Vector3ft& resulting_translation, Eigen::Matrix<double, 6,6>& resulting_covariance)
{ 
   
  ///////////////////////////////////////////////
  float timeBuildSystem = 0.f;
  float timeWarping = 0.f;
  float timeSigma = 0.f;
  float timeError = 0.f;
  float timeChiSquare = 0.f;
  float timeGradients = 0.f;
  float timeFilters = 0.f;
  float preIterLoopTime = 0.f;
  float iterLoopTime = 0.f;
  double timeBuildSystem2 = 0.f;
  double timeWarping2 = 0.f;
  
  if (real_time_flag_)
  {
    visodo_iterations_[0] = 5;
  }

  Eigen::Matrix<double, B_SIZE, B_SIZE, Eigen::RowMajor> A_total;
  Eigen::Matrix<double, B_SIZE, 1> b_total;

  float sigma_int_ref, sigma_depthinv_ref;
  sigma_int_ref = 5.f;
  sigma_depthinv_ref = 0.0025f;		
  
  float sigma_int = 40.f;
  float sigma_depthinv = 5.f;
  float bias_int = 0.f;
  float bias_depthinv = 0.f;
  float nu_int = 5.f;
  float nu_depthinv = 5.f;
  
  bool last_iter_flag = false;    
  
  float chi_test = 1.f;  
  //float chi_square_prev = 1.f;
  float chi_square = 1.f;
  float Ndof = 640.f*480.f;
  float Ndof_prev = 640.f*480.f;
  float RMSE = 9999.f;
  float RMSE_prev = 9999.f;
  
  Matrix3ft cam_rot_incremental_inv;
  Matrix3ft cam_rot_incremental;
  Vector3ft cam_trans_incremental; 
  
  Vector3f init_vector = Vector3f(0,0,0);
  float3 device_current_delta_trans;
  float3 device_current_delta_rot;  
  float3 device_inv_current_trans;
  Mat33 device_inv_current_rot; 
  
  Matrix3ft predicted_rotation;
  Vector3ft predicted_translation;
  
  Matrix3ft current_rotation;
  Vector3ft current_translation;  
  Matrix3ft inv_current_rotation;
  Vector3ft inv_current_translation;  
  
  Matrix3f inv_current_rotation_f;
  Vector3f inv_current_translation_f;
  Matrix3f current_rotation_f; 
  
  Matrix3ft previous_rotation = resulting_rotation;
  Vector3ft previous_translation = resulting_translation;
  
  //Initialize rotation and translation as if velocity was kept constant  	
  if ((global_time_ > 1) && (motion_model_ == CONSTANT_VELOCITY) && (!lost_))
  {       
    Vector3ft v_t = velocity_*delta_t_;
    Vector3ft omega_t = omega_*delta_t_;
    Matrix4ft trafo_const_vel = expMap(omega_t, v_t);
    
    Matrix3ft delta_rot_prev = trafo_const_vel.block(0,0,3,3);
    Vector3ft delta_trans_prev = trafo_const_vel.block(0,3,3,1);  

    predicted_translation = previous_rotation*delta_trans_prev + previous_translation;
    predicted_rotation = previous_rotation*delta_rot_prev;
  }
  else
  {
    predicted_translation = previous_translation;
    predicted_rotation = previous_rotation;
  } 
  
  current_rotation = predicted_rotation;
  current_translation = predicted_translation; 
  
  //last_iter_flag =  ((level_index == finest_level_) && (iter ==  (iter_num-1)));  
	
	pcl::StopWatch t4;
	
  for (int level_index = LEVELS-1; level_index>=finest_level_; --level_index)
  {    
    int iter_num = visodo_iterations_[level_index];
    
    //KF intensity and depth/inv_depth			
    IntensityMapf& intensity_odoKF = intensities_odoKF_[level_index];
    DepthMapf& depthinv_odoKF = depthinvs_odoKF_[level_index];     
    
    //We need gradients on the KF maps. 
    GradientMap& xGradInt_odoKF = xGradsInt_odoKF_[level_index];
    GradientMap& yGradInt_odoKF = yGradsInt_odoKF_[level_index];
    GradientMap& xGradDepthinv_odoKF = xGradsDepthinv_odoKF_[level_index];
    GradientMap& yGradDepthinv_odoKF = yGradsDepthinv_odoKF_[level_index];  
    MapArr& normals_odoKF = normals_odoKF_[level_index];    

    //Residuals between warped_{}_curr and {}_odoKF_
    DeviceArray<float>&  res_intensity = res_intensities_[level_index];
    DeviceArray<float>&  res_depthinv = res_depthinvs_[level_index]; 

    // run optim for iter_num iterations (return false when lost)
    for (int iter = 0; iter < iter_num; ++iter)
    {		      
      device_current_delta_trans = device_cast<float3>(init_vector);
      device_current_delta_rot = device_cast<float3>(init_vector);
      
      inv_current_rotation = current_rotation.inverse();
      inv_current_translation = - current_rotation.inverse()*current_translation;  
      
      
      current_rotation_f = current_rotation.cast<float>(); 
      
      last_iter_flag =  ((level_index == finest_level_) && (iter ==  (iter_num-1)));
      
      //device_inv_current_trans = device_cast<float3>(inv_current_translation_f);
      //device_inv_current_rot = device_cast<Mat33>(inv_current_rotation_f);       
                                       
      //first warp then pyr  
      if ((warping_ == WARP_FIRST)) //Expensive warping (warp in lvl 0, pyr down every iter)
      {    
        Eigen::Matrix3f K = getCalibMatrix(0);
        
        inv_current_rotation_f = K*inv_current_rotation.cast<float>()*K.inverse();
        inv_current_translation_f = K*inv_current_translation.cast<float>();
        
        device_inv_current_trans = device_cast<float3>(inv_current_translation_f);
        device_inv_current_rot = device_cast<Mat33>(inv_current_rotation_f);    
      
        timeWarping += warpInvDepthWithTrafo3D  (depthinvs_curr_[0], warped_depthinvs_curr_[0], depthinvs_odoKF_[0], 
                                                 device_inv_current_rot, 
                                                 device_inv_current_trans, 
                                                 cam_intrinsics(0));    
                                                                                      
        timeWarping += warpIntensityWithTrafo3DInvDepth  (intensities_curr_[0], warped_intensities_curr_[0], 
                                                          //depthinvs_odoKF_[0], 
                                                          warped_depthinvs_curr_[0], 
                                                          device_inv_current_rot, 
                                                          device_inv_current_trans, 
                                                          cam_intrinsics(0));              
        
        for (int i = 1; i < (level_index + 1); ++i)
        {     
          timeWarping+=pyrDownIntensity (warped_intensities_curr_[i-1], warped_intensities_curr_[i]);
          timeWarping+=pyrDownDepth (warped_depthinvs_curr_[i-1], warped_depthinvs_curr_[i]);	
        }
      }
      else
      {
        Eigen::Matrix3f Kscale = getCalibMatrix(level_index);
        
        inv_current_rotation_f = Kscale*inv_current_rotation.cast<float>()*Kscale.inverse();
        inv_current_translation_f = Kscale*inv_current_translation.cast<float>();
        
        device_inv_current_trans = device_cast<float3>(inv_current_translation_f);
        device_inv_current_rot = device_cast<Mat33>(inv_current_rotation_f);    
        
        timeWarping += warpInvDepthWithTrafo3D  (depthinvs_curr_[level_index], warped_depthinvs_curr_[level_index], depthinvs_odoKF_[level_index], 
                             device_inv_current_rot, 
                             device_inv_current_trans, 
                             cam_intrinsics(level_index));
                             
        timeWarping += warpIntensityWithTrafo3DInvDepth  (intensities_curr_[level_index], warped_intensities_curr_[level_index], 
                                //depthinvs_odoKF_[level_index], 
                                warped_depthinvs_curr_[level_index], 
                                device_inv_current_rot, 
                                device_inv_current_trans, 
                                cam_intrinsics(level_index));        
      }
       /////////////////////////////                                          

      IntensityMapf&  warped_intensity_curr = warped_intensities_curr_[level_index];
      DepthMapf&  warped_depthinv_curr = warped_depthinvs_curr_[level_index];     
      
      //cost function test to end iters at current pyr level
      if ((termination_ == CHI_SQUARED)&&(!(iter == 0)))
      {
        timeError += computeErrorGridStride (warped_intensities_curr_[0], intensities_odoKF_[0], res_intensities_[0]);			
        timeError += computeErrorGridStride (warped_depthinvs_curr_[0], depthinvs_odoKF_[0], res_depthinvs_[0]);
        
        timeChiSquare += computeChiSquare (res_intensities_[0], res_depthinvs_[0], sigma_int_ref, sigma_depthinv_ref, Mestimator_, chi_square, chi_test,Ndof);
        RMSE = sqrt(chi_square)/sqrt(Ndof);
        //std::cout << "lvl " << level_index << " iter " << iter << ", chiSq: " << chi_square << ",Ndof: " << Ndof << ", RMSE: "<< RMSE   << std::endl;
        
        if (!(iter == 1))
        {        
          if (RMSE > RMSE_prev) //undo the previous increment and end iters at curr pyr
          {
            current_translation = cam_rot_incremental_inv*(current_translation - cam_trans_incremental);
            current_rotation = cam_rot_incremental_inv*current_rotation;
            //std::cout << "Break in pyr " << level_index << " at iteration " << iter << std::endl; 
            break;
          }
          
          //float rel_diff = fabs(RMSE - RMSE_prev) / RMSE_prev;
          
          
          //if (rel_diff < 0.00001) //end iters at curr pyr
          //{
            ////std::cout << "Break in pyr " << level_index << " at iteration " << iter << std::endl; 
            //break;
          //}
        }   
        
        RMSE_prev = RMSE;
      }
      //////////////////////////////////
      
      
      sigma_int = 5.f;
      sigma_depthinv = 0.0025f;
      bias_int = 0.f;
      bias_depthinv = 0.f;
      nu_int = 5.f;
      nu_depthinv = 5.f;

      if (sigma_estimator_ == SIGMA_PDF)
      {
        timeError += computeErrorGridStride (warped_intensity_curr, intensity_odoKF, res_intensity, Nsamples_);			
        timeError += computeErrorGridStride (warped_depthinv_curr, depthinv_odoKF, res_depthinv, Nsamples_);
        ////timeSigma += computeDepthinvNormalsErrorGridStride (warped_depthinv_curr, depthinv_odoKF, normals_odoKF, cam_intrinsics(0), res_depthinv, Nsamples_);        
        
        timeSigma += computeSigmaAndNuStudent (res_intensity, bias_int, sigma_int, nu_int, Mestimator_);
        timeSigma +=  computeSigmaAndNuStudent(res_depthinv, bias_depthinv, sigma_depthinv, nu_depthinv, Mestimator_);
        
        //timeSigma +=  computeNuStudent(res_depthinv, bias_depthinv, sigma_depthinv, nu_depthinv);
        //timeSigma +=  computeNuStudent(res_depthinv, bias_int, sigma_int, nu_int);
        nu_int = std::max(nu_int, nu_depthinv);
        
        
        //timeSigma +=  computeSigmaPdf(res_intensity, bias_int, sigma_int, Mestimator_);
        //timeSigma +=  computeSigmaPdf(res_depthinv, bias_depthinv, sigma_depthinv, Mestimator_);
        //timeSigma +=  computeNuStudent(res_depthinv, bias_depthinv, sigma_depthinv, nu_depthinv);
        //timeSigma +=  computeNuStudent(res_depthinv, bias_int, sigma_int, nu_int);
      }
      else if  (sigma_estimator_ == SIGMA_CONS)
      {
        sigma_int = exp(log(sigma_int_ref) - 0.f*log(2)) ; //Substitute 0.f by curr lvl index??
        sigma_depthinv = exp(log(sigma_depthinv_ref) - 0.f*log(2)) ; //if depth_error_type = DEPTH, sigma_depth is not constant (depends on Z^2). As it is -> assumed const as for depth = 1m, for every depth
      }
      
      //std::cout << "Sigma int: " << sigma_int << "\t\t nu int: " << nu_int << std::endl;
      //std::cout << "sigma depthinv: " << sigma_depthinv << "\t\t nu depthinv: " << nu_depthinv << std::endl;
				
        
      //Optimise delta
      //timeBuildSystem += buildSystemGridStride (device_current_delta_trans, device_current_delta_rot,
                                      //depthinv_odoKF, intensity_odoKF,
                                      //xGradDepthinv_odoKF, yGradDepthinv_odoKF,
                                      //xGradInt_odoKF, yGradInt_odoKF,
                                      //warped_depthinv_curr, warped_intensity_curr,
                                      //Mestimator_, weighting_,
                                      //sigma_depthinv, sigma_int,
                                      //bias_depthinv, bias_int,  //there is the option of substracting the bias just in the residuals used for computing IRLS weights.
                                      //cam_intrinsics(level_index), B_SIZE,
                                      //gbuf_, sumbuf_, A_total.data (), b_total.data ());
                                      
      timeBuildSystem += buildSystemStudentNuGridStride (device_current_delta_trans, device_current_delta_rot,
                                      depthinv_odoKF, intensity_odoKF,
                                      xGradDepthinv_odoKF, yGradDepthinv_odoKF,
                                      xGradInt_odoKF, yGradInt_odoKF,
                                      warped_depthinv_curr, warped_intensity_curr,
                                      Mestimator_, weighting_,
                                      sigma_depthinv, sigma_int,
                                      bias_depthinv, bias_int,  //there is the option of substracting the bias just in the residuals used for computing IRLS weights.
                                      nu_depthinv, nu_int,
                                      cam_intrinsics(level_index), B_SIZE,
                                      gbuf_, sumbuf_, A_total.data (), b_total.data ());
                                      
                                      
      //timeBuildSystem += buildSystemWithNormalsGridStride (device_current_delta_trans, device_current_delta_rot,
                                      //depthinv_odoKF, intensity_odoKF,
                                      //normals_odoKF,
                                      //xGradDepthinv_odoKF, yGradDepthinv_odoKF,
                                      //xGradInt_odoKF, yGradInt_odoKF,
                                      //warped_depthinv_curr, warped_intensity_curr,
                                      //Mestimator_, weighting_,
                                      //sigma_depthinv, sigma_int,
                                      //bias_depthinv, bias_int,  //there is the option of substracting the bias just in the residuals used for computing IRLS weights.
                                      //cam_intrinsics(level_index), B_SIZE,
                                      //gbuf_, sumbuf_, A_total.data (), b_total.data ());
      

      MatrixXd A_optim(optim_dim_, optim_dim_);
      A_optim = A_total.block(0,0,optim_dim_,optim_dim_);

      MatrixXd b_optim(optim_dim_, 1);
      b_optim = b_total.block(0,0,optim_dim_,1);  

      MatrixXft result(optim_dim_, 1);
      result = A_optim.llt().solve(b_optim).cast<float_trafos>();

      //This part remains equal even if we used ilumination change parameters
      Vector3ft res_trans = result.block(0,0,3,1);	
      Vector3ft res_rot = result.block(3,0,3,1);	

      //If output is B_theta^A and r_B^A...
      cam_rot_incremental_inv = expMapRot(res_rot);
      cam_rot_incremental = cam_rot_incremental_inv.inverse();
      cam_trans_incremental = -cam_rot_incremental*res_trans;

      //Transform updates are applied by premultiplying. Seems counter-intuitive but it is the correct way,
      //since at each iter we warp curr frame towards prev frame.
      current_translation = cam_rot_incremental * current_translation + cam_trans_incremental;
      current_rotation = cam_rot_incremental * current_rotation;
      
      if ((std::isnan(current_rotation.norm())) || (std::isnan(current_translation.norm())))
      {
        resulting_translation = previous_translation;
        resulting_rotation = previous_rotation;
        resulting_covariance = 100.f*Eigen::Matrix<double, 6,6>::Identity();
        
        vis_odo_times_.push_back(t4.getTime());
        
        return false;
      }
      
      if (last_iter_flag) //filter gradients and compute covariance
      {
                                              
      }
    }
  } 
  
  {
    //DO the same as in a normal iteration but dont solve the system, just get the hessian inv as cov
        //IntensityMapf& intensity_odoKF_filtered = intensities_odoKF_filtered_[level_index];
        //DepthMapf& depthinv_odoKF_filtered = depthinvs_odoKF_filtered_[level_index];
        
        GradientMap& xGradInt_odoKF_covOnly = xGradsInt_odoKF_covOnly_[finest_level_];
        GradientMap& yGradInt_odoKF_covOnly = yGradsInt_odoKF_covOnly_[finest_level_];
        GradientMap& xGradDepthinv_odoKF_covOnly = xGradsDepthinv_odoKF_covOnly_[finest_level_];
        GradientMap& yGradDepthinv_odoKF_covOnly = yGradsDepthinv_odoKF_covOnly_[finest_level_]; 
        
        IntensityMapf& intensity_odoKF = intensities_odoKF_[finest_level_];
        DepthMapf& depthinv_odoKF = depthinvs_odoKF_[finest_level_];   
        DeviceArray<float>&  res_intensity = res_intensities_[finest_level_];
        DeviceArray<float>&  res_depthinv = res_depthinvs_[finest_level_]; 
        IntensityMapf&  warped_intensity_curr = warped_intensities_curr_[finest_level_];
        DepthMapf&  warped_depthinv_curr = warped_depthinvs_curr_[finest_level_]; 
        
        inv_current_rotation = current_rotation.inverse();
        inv_current_translation = - current_rotation.inverse()*current_translation;  
        
        Eigen::Matrix3f Kscale = getCalibMatrix(finest_level_);
        
        inv_current_rotation_f = Kscale*inv_current_rotation.cast<float>()*Kscale.inverse();
        inv_current_translation_f = Kscale*inv_current_translation.cast<float>();
        
        device_inv_current_trans = device_cast<float3>(inv_current_translation_f);
        device_inv_current_rot = device_cast<Mat33>(inv_current_rotation_f);    
        
        
        timeWarping += warpInvDepthWithTrafo3D  (depthinvs_curr_[finest_level_], warped_depthinvs_curr_[finest_level_], depthinvs_odoKF_[finest_level_], 
                             device_inv_current_rot, 
                             device_inv_current_trans, 
                             cam_intrinsics(finest_level_)); 
        
        timeWarping += warpIntensityWithTrafo3DInvDepth  (intensities_curr_[finest_level_], warped_intensities_curr_[finest_level_], 
                                warped_depthinvs_curr_[finest_level_], 
                                device_inv_current_rot, 
                                device_inv_current_trans, 
                                cam_intrinsics(finest_level_));

          
                             
        //timeError += computeErrorGridStride (warped_intensities_curr_[level_index], intensity_odoKF, res_intensity);			
        //timeError += computeErrorGridStride (warped_depthinvs_curr_[level_index], depthinv_odoKF, res_depthinv); 
        
        //sigma_int = 80.f;
        //sigma_depthinv = 5.5f;
        //bias_int = 0.f;
        //bias_depthinv = 0.f;
      
        
        //if (sigma_estimator_ == SIGMA_MAD)
        //{
          //timeSigma += computeSigmaMAD(res_intensity, sigma_int);
          //timeSigma += computeSigmaMAD(res_depthinv, sigma_depthinv);
        //}
        //else if (sigma_estimator_ == SIGMA_PDF)
        //{
          //timeSigma +=  computeSigmaPdf(res_intensity, bias_int, sigma_int, Mestimator_);
          //timeSigma +=  computeSigmaPdf(res_depthinv, bias_depthinv, sigma_depthinv, Mestimator_);
        //}
        //else if  (sigma_estimator_ == SIGMA_CONS)
        //{
          //sigma_int = exp(log(sigma_int_ref) - 0.f*log(2)) ; //Substitute 0.f by curr lvl index??
          //sigma_depthinv = exp(log(sigma_depthinv_ref) - 0.f*log(2)) ; 
        //}
        
        sigma_int = exp(log(sigma_int_ref) - 0.f*log(2)) ; //Substitute 0.f by curr lvl index??
        sigma_depthinv = exp(log(sigma_depthinv_ref) - 0.f*log(2)) ; 
        bias_int = 0.f;
        bias_depthinv = 0.f;
        
        //std::cout << "Sint: " << sigma_int << std::endl;
        //std::cout << "Sdepthinv: " << sigma_depthinv << std::endl;
        
        timeBuildSystem += buildSystemGridStride (device_current_delta_trans, device_current_delta_rot,
                                                  depthinv_odoKF, intensity_odoKF,
                                                  xGradDepthinv_odoKF_covOnly, yGradDepthinv_odoKF_covOnly,
                                                  xGradInt_odoKF_covOnly, yGradInt_odoKF_covOnly,
                                                  warped_depthinv_curr, warped_intensity_curr,
                                                  RGBID_SLAM::device::STUDENT, weighting_,
                                                  sigma_depthinv, sigma_int,
                                                  bias_depthinv, bias_int,  //there is the option of substracting the bias just in the residuals used for computing IRLS weights.
                                                  cam_intrinsics(finest_level_), B_SIZE,
                                                  gbuf_, sumbuf_, A_total.data (), b_total.data ());
                                                  
                                                  
          //timeBuildSystem += buildSystemWithNormalsGridStride (device_current_delta_trans, device_current_delta_rot,
                                      //depthinv_odoKF, intensity_odoKF,
                                      //normals_odoKF,
                                      //xGradDepthinv_odoKF, yGradDepthinv_odoKF,
                                      //xGradInt_odoKF, yGradInt_odoKF,
                                      //warped_depthinv_curr, warped_intensity_curr,
                                      //Mestimator_, weighting_,
                                      //sigma_depthinv, sigma_int,
                                      //bias_depthinv, bias_int,  //there is the option of substracting the bias just in the residuals used for computing IRLS weights.
                                      //cam_intrinsics(level_index), B_SIZE,
                                      //gbuf_, sumbuf_, A_total.data (), b_total.data ());
                                                  
          MatrixXd A_final(6, 6);
          A_final = A_total.block(0,0,6,6);
          
          resulting_rotation = current_rotation;
          resulting_translation = current_translation;
          
          
          //Eigen::Matrix<double,6,6> pseudo_adj = Eigen::Matrix<double,6,6>::Zero();
          //pseudo_adj.block<3,3>(0,0) = -resulting_rotation.cast<double>();
          //pseudo_adj.block<3,3>(3,3) = -resulting_rotation.cast<double>();
          
          //resulting_covariance = pseudo_adj*A_final.inverse()*pseudo_adj.transpose();   
          
          ////std::cout << "resulting_info: "<< std::endl << resulting_covariance.inverse() << std::endl;   
          //double delta_sigma_depthinv = max(0.0,(double) sigma_depthinv-sigma_depthinv_ref);
          //double sigma_sigma_depthinv = sigma_depthinv_ref/4;
          //double exp_arg_depthinv = -delta_sigma_depthinv*delta_sigma_depthinv/(2*sigma_sigma_depthinv*sigma_sigma_depthinv);
          
          //double delta_sigma_int = max(0.0,(double)sigma_int-sigma_int_ref);
          //double sigma_sigma_int = sigma_int_ref/2;
          //double exp_arg_int = -delta_sigma_int*delta_sigma_int/(2*sigma_sigma_int*sigma_sigma_int);
          
          //double cov_factor = 1.0;
          ////cov_factor = max(0.0001, exp(exp_arg_depthinv)*exp(exp_arg_int));
          
          //resulting_covariance = (1.0 / cov_factor) * resulting_covariance;
          
          resulting_covariance = A_final.inverse();   
          
          timeError += computeErrorGridStride (warped_intensities_curr_[finest_level_], intensity_odoKF, res_intensity);			
          timeError += computeErrorGridStride (warped_depthinvs_curr_[finest_level_], depthinv_odoKF, res_depthinv); 
          
          timeChiSquare += computeChiSquare (res_intensity, res_depthinv, sigma_int_ref, sigma_depthinv_ref, Mestimator_, chi_square, chi_test,Ndof);    
          RMSE = sqrt(std::max(chi_square - Ndof,0.f))/sqrt(Ndof*Ndof);
        //std::cout << "                                                                                                    Cov correction: " << (1.f / cov_factor) << std::endl;
  }
    
  std::cout << "timeTotal: " << t4.getTime() << std::endl
              << "timeTotalSum: " << timeWarping + timeBuildSystem + timeError + timeSigma + timeChiSquare + timeGradients + timeFilters<< std::endl
  						<< "  time warping: " <<  timeWarping << std::endl
  						<< "  time build system: " <<  timeBuildSystem << std::endl
  						<< "  time error: " <<  timeError << std::endl
  						<< "  time sigma: " <<  timeSigma << std::endl
  						<< "  time chiSquare: " << timeChiSquare << std::endl
  						<< "  time gradients: " << timeGradients << std::endl	
              << "  time filters: " << timeFilters << std::endl;  	
              
  //std::cout << "sigma int: "   <<  sigma_int    << std::endl
      			    //<< "sigma depthinv: " <<  sigma_depthinv  << std::endl;
  
  //Check condition number of matrix A 
  MatrixXd A_final(6, 6);
  A_final = A_total.block(0,0,6,6);
  
  MatrixXd singval_A(6, 1);
  Eigen::JacobiSVD<MatrixXd> svd(A_final);
  singval_A = svd.singularValues();
  
  double max_eig = singval_A(0,0);
  double min_eig = singval_A(0,0);
  
  for (unsigned int v_indx = 1; v_indx < 6; v_indx++)
  {
      if (singval_A(v_indx,0) >  max_eig)
        max_eig = singval_A(v_indx,0);
      if  (singval_A(v_indx,0) <  min_eig)
        min_eig = singval_A(v_indx,0);
  }
  
  double A_cn = max_eig / min_eig;
  
  //std::cout << "Matrix A condition number is: " << A_cn << std::endl;
  //std::cout << std::endl
            //<< A_final.inverse().diagonal().cwiseSqrt() << std::endl;
  
  
  //previous_rotation->_{odoKF}T^{k-1}
  //current_trafo->_{odoKF}T^curr
  //resulting_trafo->_{odoKF}T^curr (returned by estimateVisualOdometry)
  
  
  Matrix3ft delta_rot_curr = previous_rotation.transpose()*current_rotation;
  Vector3ft delta_trans_curr = previous_rotation.transpose()*(current_translation - previous_translation);
    
  Vector6ft twist = logMap(delta_rot_curr, delta_trans_curr);
  velocity_ = twist.block(0,0,3,1) * (1.f / delta_t_);
  omega_ = twist.block(3,0,3,1) * (1.f / delta_t_);  
  
  //vis_odo_times_.push_back(t4.getTime());

  //std::cout << "resulting translation: " << resulting_translation << std::endl;

  // Vis Odo has converged
  //std::cout << "RMSEt: " << sqrt(chi_square)/sqrt(Ndof) << std::endl;
  
  //return (sqrt(chi_square)/sqrt(Ndof) < 1.5f);
  return true;
}

inline float 
RGBID_SLAM::VisodoTracker::computeCovisibility(const Intr cam_intrinsics, Matrix3ft rotation_AtoB, Vector3ft translation_AtoB, 
													                          const DepthMapf& depthinvA, const DepthMapf& depthinvB)
{
	
	  float depth_tol = 0.05; //tol = 5cm (for every dpth if depth_type = depth or at 5cm at 2m if depth_type = inv_depth
	  float geom_tol = depth_tol / (2.f*2.f);
	  
	  //Check switch of odo odoKF
	  float visibility_ratio_AtoB = 1.f;
	  float visibility_ratio_BtoA = 1.f;	  
    
    Eigen::Matrix3f K = getCalibMatrix(0);
    
	  	  
	  Matrix3f rotation_AtoB_f = K*rotation_AtoB.cast<float>()*K.inverse(); //{_A}R^B
	  Vector3f translation_AtoB_f = K*translation_AtoB.cast<float>();
	  
	  Matrix3f rotation_BtoA_f = K*rotation_AtoB.inverse().cast<float>()*K.inverse();
	  Vector3f translation_BtoA_f = -K*rotation_AtoB.inverse().cast<float>()*translation_AtoB.cast<float>();
	  
	  //Visibility test to initialize new KF 
	  getVisibilityRatio(depthinvB, depthinvA, device_cast<Mat33> (rotation_AtoB_f), device_cast<float3> (translation_AtoB_f), 
						 cam_intrinsics(0), visibility_ratio_BtoA, geom_tol); 
	  //std::cout << "vis ratio: " << visibility_ratio_BtoA << std::endl;
	  
	  getVisibilityRatio(depthinvA, depthinvB, device_cast<Mat33> (rotation_BtoA_f), device_cast<float3> (translation_BtoA_f), 
						 cam_intrinsics(0), visibility_ratio_AtoB, geom_tol); 
	  //std::cout << "vis ratio: " << visibility_ratio_AtoB << std::endl;
	  
	  float visibility_ratio = std::min(visibility_ratio_AtoB, visibility_ratio_BtoA);    
	  
	  return visibility_ratio;
}


inline float 
RGBID_SLAM::VisodoTracker::computeOverlapping(const Intr cam_intrinsics, Matrix3ft rotation_AtoB, Vector3ft translation_AtoB, 
													                          const DepthMapf& depthinvA, const DepthMapf& depthinvB, BinaryMap& overlap_maskB)
{
	
	  float depth_tol = 0.05; //tol = 5cm (for every dpth if depth_type = depth or at 5cm at 2m if depth_type = inv_depth
	  float geom_tol = depth_tol / (2.f*2.f);
	  
	  //Check switch of odo odoKF
	  float visibility_ratio_BtoA = 1.f;	 
    
    Eigen::Matrix3f K = getCalibMatrix(0);
     
	  	  
	  Matrix3f rotation_AtoB_f = K*rotation_AtoB.cast<float>()*K.inverse(); //{_A}R^B
	  Vector3f translation_AtoB_f = K*translation_AtoB.cast<float>();
    
	  //Visibility test to initialize new KF 
	  getVisibilityRatioWithOverlapMask(depthinvB, depthinvA, device_cast<Mat33> (rotation_AtoB_f), device_cast<float3> (translation_AtoB_f), 
                                      cam_intrinsics(0), visibility_ratio_BtoA, geom_tol, overlap_maskB); 
	  
	  return visibility_ratio_BtoA;  
}

inline void
RGBID_SLAM::VisodoTracker::resetOdometryKeyframe()
{
  odoKF_count_ = 0;      
  
  
  //Eigen::Matrix<double,6,6> pseudo_adj_integr = Eigen::Matrix<double,6,6>::Zero();
  //pseudo_adj_integr.block<3,3>(0,0) = -delta_rotation_odo2integr_next_.transpose().cast<double>();
  //pseudo_adj_integr.block<3,3>(3,3) = -delta_rotation_odo2integr_next_.transpose().cast<double>();  

  //delta_covariance_odo2integr_next_ = delta_covariance_odo2integr_next_ + pseudo_adj_integr*delta_covariance_*pseudo_adj_integr.transpose();
  
  //T_new = T_old*T_upd
  Eigen::Matrix<double,6,6> dTnew_by_dTupd = Eigen::Matrix<double,6,6>::Zero();
  dTnew_by_dTupd.block<3,3>(0,0) = delta_rotation_odo2integr_next_.cast<double>();  
  dTnew_by_dTupd.block<3,3>(3,3) = delta_rotation_odo2integr_next_.cast<double>();  
  
  Vector3ft delta_translation_new  = delta_rotation_odo2integr_next_*delta_translation_;
  Matrix3ft skew_delta_translation_new = skew(delta_translation_new);  
  dTnew_by_dTupd.block<3,3>(0,3) = skew_delta_translation_new.cast<double>();

  delta_covariance_odo2integr_next_ = delta_covariance_odo2integr_next_ + dTnew_by_dTupd*delta_covariance_*dTnew_by_dTupd.transpose();
      
  delta_translation_odo2integr_next_ = delta_rotation_odo2integr_next_*delta_translation_ + delta_translation_odo2integr_next_;
  
  delta_rotation_odo2integr_next_ = delta_rotation_odo2integr_next_*delta_rotation_;
  
  last_odoKF_index_ = global_time_;
  last_odoKF_global_rotation_ = last_estimated_rotation_;     // [Ri|ti] - pos of camera, i.e.
  last_odoKF_global_translation_ = last_estimated_translation_;   // transform from camera to global coo space for (i-1)th camera pose
  
  delta_rotation_ = Matrix3ft::Identity ();
  delta_translation_ = Vector3ft (0, 0, 0);
  delta_covariance_ = Eigen::Matrix<double,6,6>::Zero ();
}
	
inline void
RGBID_SLAM::VisodoTracker::resetIntegrationKeyframe()
{
  integrKF_count_ = 0;
  backwards_integrKF_count_ = 0;
  //Save data of last keyframe
  rmatsKF_.push_back(last_integrKF_global_rotation_);
  tvecsKF_.push_back(last_integrKF_global_translation_);
  //std::cout << "changing kf" << std::endl;
  
  //Eigen::Matrix<double,6,6> pseudo_adj_integr = Eigen::Matrix<double,6,6>::Zero();
  //pseudo_adj_integr.block<3,3>(0,0) = -delta_rotation_odo2integr_next_.transpose().cast<double>();
  //pseudo_adj_integr.block<3,3>(3,3) = -delta_rotation_odo2integr_next_.transpose().cast<double>(); 
  //delta_covariance_odo2integr_next_ = delta_covariance_odo2integr_next_ + pseudo_adj_integr*delta_covariance_*pseudo_adj_integr.transpose();
  
  //T_new = T_old*T_upd
  Eigen::Matrix<double,6,6> dTnew_by_dTupd = Eigen::Matrix<double,6,6>::Zero();
  dTnew_by_dTupd.block<3,3>(0,0) = delta_rotation_odo2integr_next_.cast<double>();  
  dTnew_by_dTupd.block<3,3>(3,3) = delta_rotation_odo2integr_next_.cast<double>();  
  
  Vector3ft delta_translation_new  = delta_rotation_odo2integr_next_*delta_translation_;
  Matrix3ft skew_delta_translation_new = skew(delta_translation_new);  
  dTnew_by_dTupd.block<3,3>(0,3) = skew_delta_translation_new.cast<double>();

  delta_covariance_odo2integr_next_ = delta_covariance_odo2integr_next_ + dTnew_by_dTupd*delta_covariance_*dTnew_by_dTupd.transpose();
      
  
  delta_translation_odo2integr_next_ = delta_rotation_odo2integr_next_*delta_translation_ + delta_translation_odo2integr_next_;
  delta_rotation_odo2integr_next_ = delta_rotation_odo2integr_next_*delta_rotation_;
  
  //TODO: create new constraint and push_back in constraints buffer
  //Convert odo keyframe constraints (KF_last,i) to sequential constraints (i-1,i) for pose graph
  //T{k-1,k} = inv(T{odo,k-1})*T{odo,k}
  Matrix3ft delta_rotation_kf = delta_rotation_odo2integr_last_.transpose()*delta_rotation_odo2integr_next_;
  Vector3ft delta_translation_kf = delta_rotation_odo2integr_last_.transpose()*(delta_translation_odo2integr_next_ - delta_translation_odo2integr_last_);  
  
  //Eigen::Matrix<double,6,6> pseudo_adj_kf = Eigen::Matrix<double,6,6>::Zero();
  //pseudo_adj_kf.block<3,3>(0,0) = -delta_rotation_odo2integr_last_.transpose().cast<double>();
  //pseudo_adj_kf.block<3,3>(3,3) = -delta_rotation_odo2integr_last_.transpose().cast<double>();          
  //Eigen::Matrix<double,6,6> delta_covariance_kf = pseudo_adj_kf*(delta_covariance_odo2integr_next_ + delta_covariance_odo2integr_last_)*pseudo_adj_kf.transpose();  
  Eigen::Matrix<double,6,6> dDT_by_dTnew = Eigen::Matrix<double,6,6>::Zero();
  dDT_by_dTnew.block<3,3>(0,0) = delta_rotation_odo2integr_last_.transpose().cast<double>();
  dDT_by_dTnew.block<3,3>(3,3) = delta_rotation_odo2integr_last_.transpose().cast<double>();   
  
  Eigen::Matrix<double,6,6> dDT_by_dTlast = Eigen::Matrix<double,6,6>::Zero();
  dDT_by_dTlast.block<3,3>(0,0) = -delta_rotation_odo2integr_last_.transpose().cast<double>();
  dDT_by_dTlast.block<3,3>(3,3) = -delta_rotation_odo2integr_last_.transpose().cast<double>(); 
  Matrix3ft skew_delta_translation_kf = skew(delta_translation_kf);  
  dDT_by_dTlast.block<3,3>(0,3) = skew_delta_translation_kf.cast<double>()*delta_rotation_odo2integr_last_.transpose().cast<double>(); 
  
         
  Eigen::Matrix<double,6,6> delta_covariance_kf = dDT_by_dTlast*delta_covariance_odo2integr_last_*dDT_by_dTlast.transpose()+ 
                                                   dDT_by_dTnew*delta_covariance_odo2integr_next_*dDT_by_dTnew.transpose();  
  
  {          
    KeyframePtr keyframe_new_ptr(new Keyframe(getCalibMatrix(0).cast<double>(), k1_, k2_, k3_, k4_, k5_,
                                              last_integrKF_global_rotation_.cast<double>(), last_integrKF_global_translation_.cast<double>(), 
                                              delta_rotation_kf.cast<double>(),delta_translation_kf.cast<double>(),
                                              last_integrKF_index_,cols_, rows_));   
    
     
    overlap_mask_integrKF_.download(keyframe_new_ptr->overlap_mask_, cols_);
    colors_integrKF_.download(keyframe_new_ptr->colors_, cols_);
    depthinv_integrKF_.download(keyframe_new_ptr->depthinv_ , cols_);
    normals_integrKF_.download(keyframe_new_ptr->normals_, cols_);
    
    //TODO: possible blocking????
    if (keyframe_manager_ptr_->buffer_keyframes_.try_push(keyframe_new_ptr))
    {
      PoseConstraint kf_constr_new(last_integrKF_index_, global_time_, PoseConstraint::SEQ_KF, 
                                                            delta_rotation_kf.cast<double>(), delta_translation_kf.cast<double>(), 1.f, delta_covariance_kf);   
      {
        boost::mutex::scoped_lock lock(keyframe_manager_ptr_->mutex_odometry_);
        keyframe_manager_ptr_->constraints_.push_back(kf_constr_new);               
      }      
    }
    ///////////////////////
  }
  
  kf_times_.push_back(1000.f*kf_time_accum_);
  kf_time_accum_ = 0.f;
                     
  //Switch to new keyframe        
  last_integrKF_index_ = global_time_;
  
  last_integrKF_global_rotation_ = last_estimated_rotation_; 
  last_integrKF_global_translation_ = last_estimated_translation_;
  
  delta_rotation_odo2integr_last_ = delta_rotation_;
  delta_translation_odo2integr_last_ = delta_translation_;
  delta_covariance_odo2integr_last_ = delta_covariance_;  
  
  delta_rotation_odo2integr_next_ = Matrix3ft::Identity ();
  delta_translation_odo2integr_next_ = Vector3ft (0, 0, 0);
  delta_covariance_odo2integr_next_ = Eigen::Matrix<double,6,6>::Zero();
}

inline void
RGBID_SLAM::VisodoTracker::integrateImagesIntoKeyframes (DepthMapf& depthinv_src, const Intr cam_intrinsics, Matrix3ft integration_delta_rotation , Vector3ft integration_delta_translation)
{
  Matrix3ft K = getCalibMatrix(0).cast<double>();
  
  integration_delta_rotation = K*integration_delta_rotation*K.inverse();
  integration_delta_translation = K*integration_delta_translation;
  
  Matrix3ft inv_integration_delta_rotation = integration_delta_rotation.inverse();
  Vector3ft inv_integration_delta_translation = -inv_integration_delta_rotation*integration_delta_translation;
  
  Mat33 dev_inv_integration_delta_rotation;
  Mat33 dev_integration_delta_rotation;
  float3 dev_inv_integration_delta_translation;
  float3 dev_integration_delta_translation;
  
  
  {
    Matrix3f inv_integration_delta_rotation_f = inv_integration_delta_rotation.cast<float>();
    Vector3f inv_integration_delta_translation_f = inv_integration_delta_translation.cast<float>();
    
    dev_inv_integration_delta_rotation = device_cast<Mat33> (inv_integration_delta_rotation_f);
    dev_inv_integration_delta_translation = device_cast<float3> (inv_integration_delta_translation_f);
    
    //convertTransforms (integration_delta_rotation, inv_integration_delta_rotation, 
                       //inv_integration_delta_translation, integration_delta_translation,
                       //dev_integration_delta_rotation, dev_inv_integration_delta_rotation, 
                       //dev_inv_integration_delta_translation, dev_integration_delta_translation);
    
    float timeLiftAndProj = 0.f;
    float timeWarp = 0.f;
    float timeIntegration = 0.f;
    
    //Inverse warping vs...
    timeWarp += warpInvDepthWithTrafo3DWeighted  ( depthinv_src, warped_depthinv_integr_curr_, 
                                              depthinv_integrKF_, warped_weight_curr_,
                                              dev_inv_integration_delta_rotation, 
                                              dev_inv_integration_delta_translation, 
                                              cam_intrinsics(0));
        
    
    //initialiseDeviceMemory2D<float>(warped_weight_curr_, 1.f); 
    	                                         
    //timeWarp += initialiseWarpedWeights(warped_weight_curr_); 		  
                                              
    //createVMap( cam_intrinsics(0), warped_depthinv_integr_curr_, vertices_integrKF_);
    ////createVMap( cam_intrinsics(0), warped_depthinv_integr_curr_, vertices_integrKF_);
    ////createNMap( vertices_integrKF_, normals_integrKF_);   
    //computeGradientDepth(warped_depthinv_integr_curr_, xGradsDepthinv_integrKF_, yGradsDepthinv_integrKF_);  
    
    //createNMapGradients( cam_intrinsics(0), warped_depthinv_integr_curr_, xGradsDepthinv_integrKF_, yGradsDepthinv_integrKF_, normals_integrKF_);     	
    /////////////////////////////////////////////////////////////////////////////////////////////                           
                              
    //...forward warping
    //timeLiftAndProj += liftWarpAndProjInvDepth  (depthinv_src, projected_transformed_points_, depthinv_warped_in_curr_, 
                                                //dev_integration_delta_rotation, 	
                                                //dev_integration_delta_translation, 
                                                //cam_intrinsics(0));
    
    ////Visualizing warped_intensities_curr_[0], non NaN outliers appear, that corresponds to points behind an occluding area in the keyframe  
    //timeWarp += forwardMappingToWarpedInvDepthStride (depthinv_warped_in_curr_, projected_transformed_points_, 
                                                      //warped_depthinv_integr_curr_, warped_weight_curr_);	
                                                            
    //timeWarp += initialiseWarpedWeights(warped_weight_curr_);
     /////////////////////////////////////////////////////////////////////////////////////
     
                                                           
    //convertFloat2RGB (warped_intensities_curr_[0], warped_int_RGB_);
    //copyImage(intensities_curr_[0], intensity_debug_);
    
                                                            
    timeIntegration += integrateWarpedFrame(warped_depthinv_integr_curr_,  warped_weight_curr_,
                                            depthinv_integrKF_, weight_integrKF_);
                                            
    //std::cout   << "  timeLiftAndProj: " << timeLiftAndProj << std::endl
                //<< "  timeFwdWarp: " <<  timeWarp << std::endl
                //<< "  timeIntegrate: " <<  timeIntegration << std::endl;  
    
                                            
    createVMap( cam_intrinsics(0), depthinv_integrKF_, vertices_integrKF_);
    //createVMap( cam_intrinsics(0), warped_depthinv_integr_curr_, vertices_integrKF_);
    //createNMap( vertices_integrKF_, normals_integrKF_);   
    computeGradientDepth(depthinv_integrKF_, xGradsDepthinv_integrKF_, yGradsDepthinv_integrKF_);  
    
    createNMapGradients( cam_intrinsics(0), depthinv_integrKF_, xGradsDepthinv_integrKF_, yGradsDepthinv_integrKF_, normals_integrKF_);     
  }
                                            
  
  
  	
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


inline void
RGBID_SLAM::VisodoTracker::integrateCurrentRGBIntoKeyframe (Matrix3ft& integration_delta_rotation , Vector3ft& integration_delta_translation, const Intr cam_intrinsics)
{
  Matrix3ft K = getCalibMatrix(0).cast<double>();
  
  integration_delta_rotation = K*integration_delta_rotation*K.inverse();
  integration_delta_translation = K*integration_delta_translation;
  
  Matrix3ft inv_integration_delta_rotation = integration_delta_rotation.inverse();
  Vector3ft inv_integration_delta_translation = -inv_integration_delta_rotation*integration_delta_translation;
  
  Mat33 dev_inv_integration_delta_rotation;
  Mat33 dev_integration_delta_rotation;
  float3 dev_inv_integration_delta_translation;
  float3 dev_integration_delta_translation;
  
  convertTransforms (integration_delta_rotation, inv_integration_delta_rotation, 
                     inv_integration_delta_translation, integration_delta_translation,
                     dev_integration_delta_rotation, dev_inv_integration_delta_rotation, 
                     dev_inv_integration_delta_translation, dev_integration_delta_translation);
  
  float timeLiftAndProj = 0.f;
  float timeWarp = 0.f;
  float timeIntegration = 0.f;
  
  //Inverse warping vs...
  timeWarp+=warpIntensityWithTrafo3DInvDepth  (r_curr_, warped_r_curr_, 
                                                  depthinv_integrKF_, 
                                                  dev_inv_integration_delta_rotation, 
                                                  dev_inv_integration_delta_translation, 
                                                  cam_intrinsics(0) );
                                                  
  timeWarp+=warpIntensityWithTrafo3DInvDepth  (g_curr_, warped_g_curr_, 
                                                  depthinv_integrKF_, 
                                                  dev_inv_integration_delta_rotation, 
                                                  dev_inv_integration_delta_translation, 
                                                  cam_intrinsics(0) );  
                                                    
  timeWarp+=warpIntensityWithTrafo3DInvDepth  (b_curr_, warped_b_curr_, 
                                                  depthinv_integrKF_, 
                                                  dev_inv_integration_delta_rotation, 
                                                  dev_inv_integration_delta_translation, 
                                                  cam_intrinsics(0) );
                                                  
  //timeWarp += initialiseWarpedWeights(warped_weight_curr_);
  //////////////////////////////////
  
  
  //...forward warping
  //timeWarp += forwardMappingToWarpedRGBStride (depthinv_warped_in_curr_, r_curr_, g_curr_, b_curr_,
                                               //projected_transformed_points_, 
                                               //warped_depthinvs_curr_[0], warped_r_curr_, warped_g_curr_, warped_b_curr_, 
                                               //warped_weight_curr_);	
                                                          
  
                                                          
   ///////////////////////////////////////                                                       
                                                          
  timeIntegration += integrateWarpedRGB(warped_depthinvs_curr_[0], 
                                          warped_r_curr_, warped_g_curr_, warped_b_curr_,
                                          warped_weight_curr_,
                                          depthinv_integrKF_, colors_integrKF_ , weight_integrKF_);
                                          
  //std::cout   << "  timeLiftAndProj: " << timeLiftAndProj << std::endl
  						//<< "  timeFwdWarp: " <<  timeWarp << std::endl
  						//<< "  timeIntegrate: " <<  timeIntegration << std::endl; 
         		
}


///////////////////
Eigen::Affine3f
RGBID_SLAM::VisodoTracker::getCameraPose (int time) const
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  Eigen::Affine3f aff;
  aff.linear () = rmats_[time].cast<float>();
  aff.translation () = tvecs_[time].cast<float>();
  return (aff);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float
RGBID_SLAM::VisodoTracker::getVisOdoTime (int time) const
{
  if (time > (int)vis_odo_times_.size () || time < 0)
    time = vis_odo_times_.size () - 1;
  
  return (vis_odo_times_[time]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int64_t
RGBID_SLAM::VisodoTracker::getTimestamp (int time) const
{
  if (time > (int)timestamps_.size () || time < 0)
    time = timestamps_.size () - 1;
  
  return (timestamps_[time]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3d
RGBID_SLAM::VisodoTracker::getSharedCameraPose ()
{
  boost::mutex::scoped_lock lock(mutex_shared_camera_pose_);
  Eigen::Affine3d aff;
  aff = shared_camera_pose_;
  camera_pose_has_changed_ = false;
  return (aff);
}



Eigen::Matrix3f
RGBID_SLAM::VisodoTracker::getCalibMatrix (int level_index) const
{
  Eigen::Matrix3f K;
  int div = 1 << level_index; 
  
  float fx = fx_/div;
  float fy = fy_/div;
  float cx = cx_/div;
  float cy = cy_/div;
  
   K << fx,  0.f, cx,
        0.f, fy, cy,
        0.f, 0.f, 1.f;
  return K;
}



Eigen::Matrix3f
RGBID_SLAM::VisodoTracker::getCalibMatrixDepth (int level_index) const
{
  Eigen::Matrix3f K;
  int div = 1 << level_index; 
  
  float fx = fxd_/div;
  float fy = fyd_/div;
  float cx = cxd_/div;
  float cy = cyd_/div;
  
   K << fx,  0.f, cx,
        0.f, fy, cy,
        0.f, 0.f, 1.f;
  return K;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
size_t
RGBID_SLAM::VisodoTracker::getNumberOfPoses () const
{
  return rmats_.size();
}


float 
RGBID_SLAM::VisodoTracker::computeInterframeTime()
{
  float dt = 0.03333f;
  
  if (!compute_deltat_flag_)
  {
    timestamp_rgb_curr_ = 0;
    timestamp_depth_curr_ = 0;
    return dt;
  }
       
  if (global_time_  == 0)  
  {     
    uint64_t timestamp_rgb_ini = timestamp_rgb_curr_; 
    uint64_t timestamp_depth_ini = timestamp_depth_curr_; 
    
    if (timestamp_rgb_ini <= timestamp_depth_ini)
      timestamp_ini_ = timestamp_rgb_ini;
    else
      timestamp_ini_ = timestamp_depth_ini;
  }
      
  uint64_t timediff_rgb_depth = timestamp_rgb_curr_ - timestamp_depth_curr_;
  uint64_t timestamp_rgb_curr_zeroed = timestamp_rgb_curr_ - timestamp_ini_;
  uint64_t timestamp_depth_curr_zeroed = timestamp_depth_curr_ - timestamp_ini_; 
  
  timestamps_.push_back(timestamp_rgb_curr_zeroed);
  
  if (global_time_  > 0)    
  {
    dt = (float)(1e-9*(timestamp_rgb_curr_zeroed - (timestamps_[global_time_-1])));      
    std::cout << "Timestamp diff to prev: " << dt << std::endl;
  }
  return dt;
}
  
       
bool
RGBID_SLAM::VisodoTracker::trackNewFrame()
{  
  // Intrisics of the camera    
    delta_t_ = computeInterframeTime();
    //delta_t_ = 0.033333;
    kf_time_accum_ += delta_t_;
    Intr intr (fx_, fy_, cx_, cy_, k1_, k2_, k3_, k4_, k5_);
    Intr intr_depth (fxd_, fyd_, cxd_, cyd_, k1d_, k2d_, k3d_, k4d_, k5d_);
    DepthDist depth_dist (c1_, c0_, alpha0_, alpha1_, alpha2_, kspd1_, kspd2_, kspd3_, kspd4_, kspd5_, kspd6_, kspd7_, kspd8_);

    pcl::ScopeTime t1 ("whole loop");
    {
      //pcl::ScopeTime t2 ("prepare images");
      if (custom_registration_)
      {
        prepareImagesCustomCalibration(depth_, rgb24_, intr, intr_depth, depth_dist);
      }
      else
      {
        prepareImages (depth_, rgb24_, intr); 
      }
    }
    // sync GPU device
    RGBID_SLAM::device::sync ();
    
    if (global_time_ == 0) 
    {        
      ++global_time_;
      
      odoKF_count_ = 0;
      last_odoKF_index_ = 0;      
      last_odoKF_global_rotation_ = rmats_[global_time_ - 1];     // [Ri|ti] - pos of camera, i.e.
      last_odoKF_global_translation_ = tvecs_[global_time_ - 1];   // transform from camera to global coo space for (i-1)th camera pose
      
      integrKF_count_ = 0;
      backwards_integrKF_count_ = 0;
      last_integrKF_index_ = 0;      
      last_integrKF_global_rotation_ = Matrix3ft::Identity ();
      last_integrKF_global_translation_ = Vector3ft (0, 0, 0);
      
      delta_rotation_ = Matrix3ft::Identity ();
      delta_translation_ = Vector3ft (0, 0, 0); 
      delta_covariance_ = Eigen::Matrix<double,6,6>::Zero ();   
      
      odo_rmats_.push_back(delta_rotation_);
      odo_tvecs_.push_back(delta_translation_);
      odo_covmats_.push_back(delta_covariance_);   
      
      delta_rotation_odo2integr_last_ = Matrix3ft::Identity ();
      delta_translation_odo2integr_last_ = Vector3ft (0, 0, 0); 
      delta_covariance_odo2integr_last_ = Eigen::Matrix<double,6,6>::Zero ();
      
      delta_rotation_odo2integr_next_ = Matrix3ft::Identity ();
      delta_translation_odo2integr_next_ = Vector3ft (0, 0, 0);
      delta_covariance_odo2integr_next_ = Eigen::Matrix<double,6,6>::Zero ();
      
      saveCurrentImagesAsOdoKeyframes (intr); 
      saveCurrentImagesAsIntegrationKeyframes (intr, rgb24_); 
      initialiseDeviceMemory2D<unsigned char> (overlap_mask_integrKF_, 0);
      
      //keyframe_times_.push_back(0.f);
      kf_time_accum_ = 0.f;
      
      {
        Pose pose_new(0, Eigen::Matrix3d::Identity (), Eigen::Vector3d (0, 0, 0));
        
        {
          boost::mutex::scoped_lock lock(keyframe_manager_ptr_->mutex_odometry_);
          keyframe_manager_ptr_->poses_.push_back(pose_new); 
        }
        
        setSharedCameraPose(pose_new.getAffine());
      }
      
      // return and wait for next frame
      return (false);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////

    Matrix3ft delta_rotation_prev = delta_rotation_;
    Vector3ft delta_translation_prev = delta_translation_;  
    Eigen::Matrix<double,6,6> delta_covariance_prev = delta_covariance_;
    
    //Daniel: visual odometry by photometric minimisation in I and D
    RGBID_SLAM::device::sync ();  
    
    if (lost_ == false)
    {
      //pcl::ScopeTime t_odo ("odo estimation");
      odometry_success_ = estimateVisualOdometry(intr, delta_rotation_, delta_translation_, delta_covariance_);
      
      last_estimated_translation_ = last_odoKF_global_translation_ + last_odoKF_global_rotation_ * delta_translation_;
      last_estimated_rotation_ = last_odoKF_global_rotation_ * delta_rotation_;
      rmats_.push_back (last_estimated_rotation_); 
      tvecs_.push_back (last_estimated_translation_);
      
      if  (!odometry_success_) //in this case delta_rotation_ and delta_translation_ have no been updated
      { 
        lost_ = true;
        //Introduce dummy odometry constraint (zero motion, very high covariance)
        {
          PoseConstraint odo_constr_new(global_time_-1, global_time_, PoseConstraint::SEQ_ODO, 
                                        Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero(), 1.f, 100.f*Eigen::Matrix<double, 6, 6>::Identity());
          Pose pose_new(global_time_, last_estimated_rotation_.cast<double>(), last_estimated_translation_.cast<double>());
          {
            boost::mutex::scoped_lock lock(keyframe_manager_ptr_->mutex_odometry_);
            keyframe_manager_ptr_->constraints_.push_back(odo_constr_new); 
            pose_new.rotation_ = (keyframe_manager_ptr_->poses_.back().rotation_) ;
            pose_new.translation_ = keyframe_manager_ptr_->poses_.back().translation_;
            keyframe_manager_ptr_->poses_.push_back(pose_new); 
          }
        }
        
        //Reset odometry and integration keyframes. 
        resetOdometryKeyframe();
        resetIntegrationKeyframe(); //Includes buffering last keyframe before odo failure and dummy seq kf constraint
        
        saveCurrentImagesAsOdoKeyframes(intr);
        saveCurrentImagesAsIntegrationKeyframes(intr, rgb24_); 
        
        push_depthinv_buffer_ptr->clear();
        pop_depthinv_buffer_ptr->clear();
        
        ++global_time_;
        
        std::cout << "I am LOST!!!" << std::endl;
        return false;
      } 
    }
    else
    {
      odometry_success_ = estimateVisualOdometry(intr, delta_rotation_, delta_translation_, delta_covariance_);
      
      if (odometry_success_)
      {
        lost_ = false;
        last_estimated_translation_ = last_odoKF_global_translation_ + last_odoKF_global_rotation_ * delta_translation_;
        last_estimated_rotation_ = last_odoKF_global_rotation_ * delta_rotation_;
        rmats_.push_back (last_estimated_rotation_); 
        tvecs_.push_back (last_estimated_translation_);
      }      
      else
      {
        saveCurrentImagesAsOdoKeyframes(intr);
        saveCurrentImagesAsIntegrationKeyframes(intr, rgb24_); 
        return false;        
      }
    }
      
    RGBID_SLAM::device::sync ();

    // Save newly-computed pose
    
    
    odoKF_count_ ++;
    integrKF_count_++;    
    
    //Convert odo keyframe constraints (KF_last,i) to sequential constraints (i-1,i) for pose graph
    Matrix3ft delta_rotation_seq = delta_rotation_prev.transpose()*delta_rotation_;
    Vector3ft delta_translation_seq = delta_rotation_prev.transpose()*(delta_translation_ - delta_translation_prev); 
       
    //Eigen::Matrix<double,6,6> pseudo_adj_odo = Eigen::Matrix<double,6,6>::Zero();
    //pseudo_adj_odo.block<3,3>(0,0) = -delta_rotation_prev.transpose().cast<double>();
    //pseudo_adj_odo.block<3,3>(3,3) = -delta_rotation_prev.transpose().cast<double>();          
    //Eigen::Matrix<double,6,6> delta_covariance_seq = pseudo_adj_odo*(delta_covariance_ + delta_covariance_prev)*pseudo_adj_odo.transpose();
    Eigen::Matrix<double,6,6> dDT_by_dTnew = Eigen::Matrix<double,6,6>::Zero();
    dDT_by_dTnew.block<3,3>(0,0) = delta_rotation_prev.transpose().cast<double>();
    dDT_by_dTnew.block<3,3>(3,3) = delta_rotation_prev.transpose().cast<double>();   
    
    Eigen::Matrix<double,6,6> dDT_by_dTlast = Eigen::Matrix<double,6,6>::Zero();
    dDT_by_dTlast.block<3,3>(0,0) = -delta_rotation_prev.transpose().cast<double>();
    dDT_by_dTlast.block<3,3>(3,3) = -delta_rotation_prev.transpose().cast<double>(); 
    Matrix3ft skew_delta_translation_seq = skew(delta_translation_seq);  
    dDT_by_dTlast.block<3,3>(0,3) = skew_delta_translation_seq.cast<double>()*delta_rotation_prev.transpose().cast<double>(); 
    
           
    Eigen::Matrix<double,6,6> delta_covariance_seq = dDT_by_dTlast*delta_covariance_prev*dDT_by_dTlast.transpose()+ 
                                                     dDT_by_dTnew*delta_covariance_*dDT_by_dTnew.transpose(); 
                                                   
    
    odo_rmats_.push_back(delta_rotation_seq);
    odo_tvecs_.push_back(delta_translation_seq);
    odo_covmats_.push_back(delta_covariance_seq);
        
    {
      PoseConstraint odo_constr_new(global_time_-1, global_time_, PoseConstraint::SEQ_ODO, 
                                                              delta_rotation_seq.cast<double>(), delta_translation_seq.cast<double>(), 1.f, delta_covariance_seq);
      Pose pose_new(global_time_, last_estimated_rotation_.cast<double>(), last_estimated_translation_.cast<double>());
      {
        boost::mutex::scoped_lock lock(keyframe_manager_ptr_->mutex_odometry_);
        keyframe_manager_ptr_->constraints_.push_back(odo_constr_new); 
        pose_new.rotation_ = (keyframe_manager_ptr_->poses_.back().rotation_) * delta_rotation_seq.cast<double>();
        pose_new.translation_ = keyframe_manager_ptr_->poses_.back().translation_ + (keyframe_manager_ptr_->poses_.back().rotation_) * delta_translation_seq.cast<double>();
        keyframe_manager_ptr_->poses_.push_back(pose_new); 
        keyframe_manager_ptr_->trajectory_has_changed_ = true;
      }
      
      setSharedCameraPose (pose_new.getAffine());
    }
    ///////////////////////////////////////////////////////////////
    
    //Visibility checks and intgrations better here!!  
    float visibility_ratio_odo = computeCovisibility(intr, delta_rotation_, delta_translation_, 
                                                     depthinvs_odoKF_[0], depthinvs_curr_[0]);
                       
    if (( odoKF_count_ >= max_odoKF_count_ ) || (visibility_ratio_odo < visibility_ratio_odo_threshold_)) //include lost condition and visibility condition
    {
      resetOdometryKeyframe();
    
      saveCurrentImagesAsOdoKeyframes (intr); 
    }  
    
    Matrix3ft delta_integr_rotation = last_integrKF_global_rotation_.inverse()*last_estimated_rotation_;
    Vector3ft delta_integr_translation = last_integrKF_global_rotation_.inverse()*(last_estimated_translation_ - last_integrKF_global_translation_);
    
    float visibility_ratio_integr = computeCovisibility(intr, delta_integr_rotation, delta_integr_translation, 
                                                        depthinv_integrKF_raw_, depthinvs_curr_[0]);
                                                        
    //std::cout << delta_integr_rotation << std::endl;
    //std::cout << delta_integr_translation << std::endl;
    //std::cout << "visibility ratio: " << visibility_ratio_integr << std::endl;
    {
      //pcl::ScopeTime t_kf ("keyframe integration");
      
      
      if ( ( integrKF_count_ >= max_integrKF_count_ ) || (visibility_ratio_integr < visibility_ratio_integr_threshold_) )
      {	   
        resetIntegrationKeyframe();   
      
        computeOverlapping(intr, delta_integr_rotation, delta_integr_translation, 
                   depthinv_integrKF_raw_, depthinvs_curr_[0], overlap_mask_integrKF_);
                   
        saveCurrentImagesAsIntegrationKeyframes (intr, rgb24_); 
        
        swapBufferPointers();
        push_depthinv_buffer_ptr->clear();
        
        if (!pop_depthinv_buffer_ptr->empty())
          pop_depthinv_buffer_ptr->pop_back();          
      }
      else
      {
        //std::cout << "integrating kf" << std::endl;
        integrateImagesIntoKeyframes (depthinvs_curr_[0], intr, delta_integr_rotation, delta_integr_translation);
        //integrateCurrentRGBIntoKeyframe(delta_integr_rotation, delta_integr_translation, intr);
      }
    
      if (!pop_depthinv_buffer_ptr->empty()) 
      {        
        Matrix3ft delta_back_integr_rotation = last_integrKF_global_rotation_.inverse()*rmats_[global_time_-integrKF_count_-backwards_integrKF_count_-1];
        Vector3ft delta_back_integr_translation = last_integrKF_global_rotation_.inverse()*(tvecs_[global_time_-integrKF_count_-backwards_integrKF_count_-1] - last_integrKF_global_translation_);
        
        integrateImagesIntoKeyframes ((pop_depthinv_buffer_ptr->back()), intr, delta_back_integr_rotation, delta_back_integr_translation); 
                     
        pop_depthinv_buffer_ptr->pop_back();  
        backwards_integrKF_count_++; 
      } 
      
      //saveIntegrationKeyframesAsOdoKeyframes(intr);     
    }
    
    vis_odo_times_.push_back(t1.getTime());
    
    {
      boost::mutex::scoped_lock lock(mutex_scene_view_);
      
      getImage(scene_view_, intensity_view_, depthinv_view_);
      scene_view_has_changed_ = true;      
    }
    
    
    
    RGBID_SLAM::device::sync ();
    
    ++global_time_;
    
    return(true);
}

bool
RGBID_SLAM::VisodoTracker::operator() ()
{ 
  boost::unique_lock<boost::mutex> lock(mutex_);
  exit_ = false;
  
  {
    boost::mutex::scoped_try_lock lock(created_aux_mutex_);
    created_cond_.notify_one();
  }
  
  while (!exit_)
  {
    new_frame_cond_.wait (lock);
    trackNewFrame();   
  }

  return (true);
}




