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

#ifndef VISODO_HPP_
#define VISODO_HPP_

//#include <pcl/pcl_macros.h>
//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>
//#include <pcl/io/ply_io.h>

#include <Eigen/Core>
#include <vector>
#include <boost/thread/thread.hpp>

#include "internal.h"

#include "float3_operations.h"
#include "../ThirdParty/pcl_gpu_containers/include/device_array.h"
#include "types.h"
#include "settings.h"



namespace RGBID_SLAM
{
        
    /** \brief VisodoTracker class encapsulates dense RGBD visual odometry implementation
      */
      
    class VisodoTracker
    {
      public:
        
        /** \brief Number of pyramid levels */
        enum { LEVELS = 3 }; 
        
        VisodoTracker (int optim_dim = 6, 
                       int Mestimator = RGBID_SLAM::device::DEFAULT_MESTIMATOR,
                       int motion_model = RGBID_SLAM::device::DEFAULT_MOTION_MODEL,
                       int sigma_estimator = RGBID_SLAM::device::SIGMA_PDF, 
                       int weighting = RGBID_SLAM::device::DEFAULT_WEIGHTING,
                       int warping = RGBID_SLAM::device::WARP_FIRST, 
                       int max_odoKF_count = RGBID_SLAM::device::DEFAULT_ODO_KF_COUNT,
                       int finest_level = 0,
                       int termination = RGBID_SLAM::device::DEFAULT_TERMINATION, 
                       float visratio_odo = RGBID_SLAM::device::DEFAULT_VISRATIO_ODO, 
                       int image_filtering = RGBID_SLAM::device::DEFAULT_IMAGE_FILTERING, 
                       float visratio_integr = RGBID_SLAM::device::DEFAULT_VISRATIO_INTEGR,
                       int max_integrKF_count = RGBID_SLAM::device::DEFAULT_INTEGR_KF_COUNT,                        
                       int Nsamples = 10000,
                       int rows = 480, int cols = 640);
                       
        void loadSettings (Settings &settings);
                       
        void loadCalibration(std::string const &calib_file);    
        
        void start();
        
        bool trackNewFrame();        
        
        int  cols ();        
        int rows ();

        bool operator() ();  
        
        void setRGBIntrinsics (float fx, float fy, float cx = -1, float cy = -1, 
                                float k1 = 0.f, float k2 = 0.f, float k3 = 0.f, float k4 = 0.f, float k5 = 0.f );
       
                                  
        void setDepthIntrinsics (float fxd, float fyd, float cxd = -1, float cyd = -1, 
                                  float k1d = 0.f, float k2d = 0.f, float k3d = 0.f, float k4d = 0.f, float k5d = 0.f,
                                  float c0 = 0.f, float c1 = 1.f, float alpha0 = 1.f, float alpha1 = 0.f, float alpha2 = 0.f, 
                                  float kspd1 = 0.f, float kspd2 = 0.f, float kspd3 = 0.f, float kspd4 = 0.f, 
                                  float kspd5 = 0.f, float kspd6 = 0.f, float kspd7 = 0.f, float kspd8 = 0.f);
                                
        void setSharedCameraPose (const Eigen::Affine3d& pose);       

        Eigen::Affine3f getCameraPose (int time = -1) const;                
        float getVisOdoTime (int time = -1) const;        
        int64_t getTimestamp (int time = -1) const;   
             
        Eigen::Affine3d getSharedCameraPose ();
        
        size_t  getNumberOfPoses () const;  
              
        Eigen::Matrix3f getCalibMatrix (int level_index=0) const;  
        Eigen::Matrix3f getCalibMatrixDepth (int level_index=0) const;  
              
        void  getImage (std::vector<PixelRGB>& scene_view, std::vector<float>& depthinv_view, std::vector<float>& intensity_view );        
        
        bool visOdoIsLost ()  { return (lost_);}        
        
        void  reset ();

        bool newKF_;
        std::vector<Eigen::Matrix3d> rmatsKF_;
        std::vector<Eigen::Vector3d> tvecsKF_;        
        
        View rgb24_;  
        DepthMap depth_;
        uint64_t timestamp_rgb_curr_;
        uint64_t timestamp_depth_curr_;
        uint64_t timestamp_ini_;
        
        boost::mutex mutex_;
        boost::condition_variable new_frame_cond_;
        boost::mutex created_aux_mutex_;
        boost::condition_variable created_cond_;
        
        bool compute_deltat_flag_;
        bool real_time_flag_;
        
        bool exit_;
        
        KeyframeManagerPtr keyframe_manager_ptr_;
        
        std::vector<float> kf_times_;
        
        boost::mutex mutex_shared_camera_pose_;
        Eigen::Affine3d shared_camera_pose_;
        bool camera_pose_has_changed_;
        bool odometry_success_;
        
        boost::mutex mutex_scene_view_;
        std::vector<PixelRGB> scene_view_;
        std::vector<float> intensity_view_;
        std::vector<float> depthinv_view_;
        bool scene_view_has_changed_;
      
      private:
      
        float 
        computeInterframeTime();        
        
        void allocateBuffers (int rows_arg, int cols_arg);
        
        inline void convertTransforms (Matrix3ft& transform_in_1, Matrix3ft& transform_in_2, Vector3ft& translation_in_1, Vector3ft& translation_in_2, RGBID_SLAM::device::Mat33& transform_out_1, RGBID_SLAM::device::Mat33& transform_out_2, float3& translation_out_1, float3& translation_out_2);        
        
        inline void convertTransforms (Matrix3ft& transform_in_1, Matrix3ft& transform_in_2, Vector3ft& translation_in,
                           RGBID_SLAM::device::Mat33& transform_out_1, RGBID_SLAM::device::Mat33& transform_out_2, float3& translation_out);        
       
        inline void convertTransforms (Matrix3ft& transform_in, Vector3ft& translation_in,
                           RGBID_SLAM::device::Mat33& transform_out, float3& translation_out);

        //Debugging functions see what is happening with warped or integrated generated images
        void downloadAndSaveIntensity(const DeviceArray2D<float>& dev_image, int pyrlevel = 0, int iterat = 0);

        void downloadAndSaveDepth(const DeviceArray2D<float>& dev_image, int pyrlevel = 0, int iterat = 0);
        //////////////////////////////
  
        
        inline void prepareImages (const DepthMap& depth_raw, const View& colors_raw, const RGBID_SLAM::device::Intr& cam_intrinsics);
        
        inline void prepareImagesCustomCalibration (const DepthMap& depth_raw, const View& colors_raw, 
                                                              const RGBID_SLAM::device::Intr& rgb_intrinsics, const RGBID_SLAM::device::Intr& depth_intrinsics, 
                                                              const RGBID_SLAM::device::DepthDist& depth_spdist);
        
        inline bool estimateVisualOdometry(const RGBID_SLAM::device::Intr cam_intrinsics, Matrix3ft& resulting_rotation , Vector3ft& resulting_translation, Eigen::Matrix<double,6,6>& resulting_covariance);          

        inline float computeCovisibility(const RGBID_SLAM::device::Intr cam_intrinsics, Matrix3ft rotation_AtoB, Vector3ft translation_AtoB, 
													const DepthMapf& depthinvA, const DepthMapf& depthinvB);
                          
        inline float computeOverlapping(const RGBID_SLAM::device::Intr cam_intrinsics, Matrix3ft rotation_AtoB, Vector3ft translation_AtoB, 
													const DepthMapf& depthinvA, const DepthMapf& depthinvB, BinaryMap& overlap_maskB);
                
        inline void resetOdometryKeyframe();
        
        inline void resetIntegrationKeyframe();
        
        inline void integrateImagesIntoKeyframes (DepthMapf& depthinv_src, const RGBID_SLAM::device::Intr cam_intrinsics, Matrix3ft delta_rotation , Vector3ft delta_translation);
        
        inline void integrateCurrentRGBIntoKeyframe (Matrix3ft& integration_delta_rotation , Vector3ft& integration_delta_translation, const RGBID_SLAM::device::Intr cam_intrinsics);
        
        inline void saveCurrentImages();	
        inline void saveCurrentImagesAsOdoKeyframes(const RGBID_SLAM::device::Intr cam_intrinsics);	
        inline void savePreviousImagesAsOdoKeyframes();	        
        inline void saveIntegrationKeyframesAsOdoKeyframes(const RGBID_SLAM::device::Intr cam_intrinsics);	        
        inline void saveCurrentImagesAsIntegrationKeyframes(const RGBID_SLAM::device::Intr cam_intrinsics, const View& colors);	
        
        inline void initialiseBufferPointers();
        inline void swapBufferPointers();    
        
        //void downloadAndSaveDepth(const DeviceArray2D<float>& dev_image, int pyrlevel = 0, int iterat = 0);
          
        int rows_;
        int cols_;
        int global_time_;
        
        //TODO: read from file
        float fx_, fy_, cx_, cy_, k1_, k2_, k3_, k4_, k5_, factor_depth_;
        float fxd_, fyd_, cxd_, cyd_, k1d_, k2d_, k3d_, k4d_, k5d_;
        int custom_registration_;
        
        float c1_, c0_, alpha0_, alpha1_, alpha2_, kspd1_, kspd2_, kspd3_, kspd4_, kspd5_, kspd6_, kspd7_, kspd8_;
        int x_shift_;
        int y_shift_;
        
        Matrix3f dRc_;
        Vector3f t_dc_;
        
        Matrix3ft init_Rcam_;
        Vector3ft   init_tcam_;

        int visodo_iterations_[LEVELS];
        
        std::vector<DepthMapf> depthinvs_curr_;  
        std::vector<IntensityMapf> intensities_curr_;
        std::vector<DepthMapf> depthinvs_prev_;
        std::vector<IntensityMapf> intensities_prev_;
        
        std::vector<DepthMapf> depthinvs_odoKF_;
        std::vector<IntensityMapf> intensities_odoKF_;
        std::vector<DepthMapf> depthinvs_odoKF_filtered_;
        std::vector<IntensityMapf> intensities_odoKF_filtered_;

        DepthMapf depthinv_distorted_;
        DepthMapf depthinv_corr_distorted_;
        IntensityMapf intensity_distorted_;
        DepthMapf depthinv_preregister_;
        DepthMapf depthinv_register_trans_;
        DeviceArray2D<int> depthinv_register_trans_as_int_;
        
        DepthMapf depthinv_integrKF_;
        IntensityMapf intensity_integrKF_;
        DepthMapf depthinv_integrKF_raw_;
        IntensityMapf intensity_integrKF_raw_;
        View colors_integrKF_;
        BinaryMap overlap_mask_integrKF_;
        
        std::vector<DepthMapf> depthinv_integrKF_pyr_;
        
        IntensityMapf r_curr_;
        IntensityMapf b_curr_;
        IntensityMapf g_curr_;
        
        IntensityMapf warped_r_curr_;
        IntensityMapf warped_b_curr_;
        IntensityMapf warped_g_curr_;
        
        
        std::vector<GradientMap> xGradsInt_odoKF_;
        std::vector<GradientMap> yGradsInt_odoKF_;
        std::vector<GradientMap> xGradsDepthinv_odoKF_;
        std::vector<GradientMap> yGradsDepthinv_odoKF_;
        
        std::vector<MapArr> vertices_odoKF_;
        std::vector<MapArr> normals_odoKF_;
        
        std::vector<GradientMap> xGradsInt_odoKF_covOnly_;
        std::vector<GradientMap> yGradsInt_odoKF_covOnly_;
        std::vector<GradientMap> xGradsDepthinv_odoKF_covOnly_;
        std::vector<GradientMap> yGradsDepthinv_odoKF_covOnly_;


        std::vector<DepthMapf> warped_depthinvs_curr_;
        std::vector<IntensityMapf> warped_intensities_curr_;
        IntensityMapf intensity_debug_;
        View warped_int_RGB_;
        
        DeviceArray2D<Point2D> rect_corner_pos_;
        
        DepthMapf warped_depthinv_integr_curr_;
        IntensityMapf warped_intensity_integr_curr_;
        
        std::vector< DeviceArray<float> > res_intensities_;
        std::vector< DeviceArray<float> > res_depthinvs_;
        
        DeviceArray2D<float> projected_transformed_points_;
        DeviceArray2D<float> depthinv_warped_in_curr_; 
        DeviceArray2D<float> warped_weight_curr_;
        DeviceArray2D<float> weight_integrKF_;
        
        std::deque<DeviceArray2D<float> > depthinv_buffer1;
        std::deque<DeviceArray2D<float> > depthinv_buffer2;
        std::deque<DeviceArray2D<float> >* pop_depthinv_buffer_ptr;
        std::deque<DeviceArray2D<float> >* push_depthinv_buffer_ptr;
        
        
        MapArr vertices_integrKF_;
        MapArr normals_integrKF_;
        
        GradientMap xGradsDepthinv_integrKF_;
        GradientMap yGradsDepthinv_integrKF_;
        
        std::vector<MapArr> vertices_integrKF_pyr_;
        std::vector<MapArr> normals_integrKF_pyr_;
        
        /** \brief Temporary buffer for tracking */
        DeviceArray2D<RGBID_SLAM::device::float_type> gbuf_;

        /** \brief Buffer to store MLS matrix. */
        DeviceArray<RGBID_SLAM::device::float_type> sumbuf_;

        /** \brief Array of camera rotation matrices for each moment of time. */
        std::vector<Matrix3ft> rmats_;

        /** \brief Array of camera translations for each moment of time. */
        std::vector<Vector3ft> tvecs_;
        
        /** \brief Array of odometry rotation matrices for each moment of time. */
        std::vector<Matrix3ft> odo_rmats_;

        /** \brief Array of odometry translations for each moment of time. */
        std::vector<Vector3ft> odo_tvecs_;
        
        /** \brief Array of odometry covariance for each moment of time. */
        std::vector< Eigen::Matrix<double,6,6> > odo_covmats_;        
        
        /** \brief Array of chi tests for each moment of time. */
        std::vector<float> chi_tests_;
        
        /** \brief Array of visual odometry time costs for each moment of time. */
        std::vector<float> vis_odo_times_;
        
        /** \brief Array of timestamps for each moment of time. */
        std::vector<int64_t> timestamps_;

        /** \brief True if tracker is lost */
        bool lost_;
        
        Matrix3ft last_estimated_rotation_;
        Vector3ft last_estimated_translation_;

        int odoKF_count_;
        int integrKF_count_;
        int backwards_integrKF_count_;
        int last_odoKF_index_;
        int last_integrKF_index_;
             
        float delta_t_;
        Vector3ft velocity_;
        Vector3ft omega_;
	
	      int optim_dim_;  //not used
        int Mestimator_;
        int motion_model_;
        int sigma_estimator_;
        int weighting_;
        int warping_;
        int max_odoKF_count_;
        int finest_level_;
        int termination_;
        float visibility_ratio_odo_threshold_;
        int image_filtering_;
        float visibility_ratio_integr_threshold_; 
        int max_integrKF_count_;        
        int Nsamples_;
        int num_prev_integr_;
        
        
        
        boost::shared_ptr<boost::thread> visodo_thread_;
        
        // Update Pose
        Matrix3ft delta_rotation_;
        Vector3ft delta_translation_;
        Eigen::Matrix<double,6,6> delta_covariance_;
        Matrix3ft last_odoKF_global_rotation_;
        Vector3ft last_odoKF_global_translation_;
        Matrix3ft last_integrKF_global_rotation_;
        Vector3ft last_integrKF_global_translation_;
        
        Matrix3ft delta_rotation_odo2integr_last_;
        Vector3ft delta_translation_odo2integr_last_; 
        Eigen::Matrix<double,6,6> delta_covariance_odo2integr_last_;
        
        Matrix3ft delta_rotation_odo2integr_next_;
        Vector3ft delta_translation_odo2integr_next_;
        Eigen::Matrix<double,6,6> delta_covariance_odo2integr_next_;
        
        float kf_time_accum_;
        
        public:
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

#endif /* VISODO_HPP_ */
