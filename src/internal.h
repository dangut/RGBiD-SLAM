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

#ifndef RGBIDSLAM_INTERNAL_HPP_
#define RGBIDSLAM_INTERNAL_HPP_

//#include <pcl/gpu/containers/device_array.h>
#include "../ThirdParty/pcl_gpu_containers/include/device_array.h"
//#include "types.h"
#include "../ThirdParty/pcl_gpu_containers/include/impl/safe_call.hpp"

#include <iostream> // used by operator << in Struct Intr

using namespace pcl::gpu;

namespace RGBID_SLAM
{
  namespace device
  {
    extern cudaDeviceProp dev_prop;
    extern int dev_id;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Types
    typedef unsigned short ushort;
    typedef unsigned char uchar;
    typedef DeviceArray2D<float> MapArr;
    typedef DeviceArray2D<ushort> DepthMap;
    typedef DeviceArray2D<uchar> IntensityMap;
    typedef DeviceArray2D<float> DepthMapf;
    typedef DeviceArray2D<float> IntensityMapf;
    typedef DeviceArray2D<float> GradientMap;
    typedef DeviceArray2D<uchar> BinaryMap;
    typedef float4 PointType;
    typedef double float_type;

    //TSDF fixed point divisor (if old format is enabled)
    const int DIVISOR = 32767;     // SHRT_MAX;

    //RGB images resolution
    const float  HEIGHT = 480.0f;
    const float  WIDTH = 640.0f;
    
    enum 
    {
      B_SIZE = 6,
      A_SIZE = (B_SIZE * B_SIZE - B_SIZE) / 2 + B_SIZE,
      TOTAL_SIZE = A_SIZE + B_SIZE
    };

    enum {LSQ, HUBER, TUKEY, STUDENT};
    enum {NO_MM, CONSTANT_VELOCITY};		
    enum {SIGMA_MAD, SIGMA_PDF, SIGMA_CONS};	
    enum {INDEPENDENT, MIN_WEIGHT, GEOM_ONLY, PHOT_ONLY};
    enum {WARP_FIRST, PYR_FIRST};
    enum {CHI_SQUARED, ALL_ITERS};
    enum {NO_FILTERS, FILTER_GRADS};
    
    const float THRESHOLD_HUBER = 1.345f;
    const float THRESHOLD_TUKEY = 4.685f;
    const float STUDENT_DOF = 5.f;

    //Temporary constant (until we make it automatic) that holds the Kinect's focal length
    //const float FOCAL_LENGTH = 575.816f;

    //xtion Bristol
    //      const float FOCAL_LENGTH = 540.60f;
    //      const float CENTER_X = 317.76f;
    //      const float CENTER_Y = 240.76f;

    //xtion Daniel's
    const float FOCAL_LENGTH = 543.78f;
    const float CENTER_X = 313.45f;
    const float CENTER_Y = 235.00f;
    
    const float FOCAL_LENGTH_DEPTH = 580.f;

    const float BASELINE = 75.0f; //in mm. Not an exact calibration. Just coarse approx by a rule.

    //Constants for checking vOdo pyramid iterations termination
    const float MIN_DEPTH = 0.5f;  //real one might be a little more, but better to overestimate
    const float PIXEL_ACC = 0.5f; //pixel error, should put less, to be a little bit conservative??
    
    const int SNAPSHOT_RATE = 45; // every 45 frames an RGB snapshot will be saved.

   
    const int DEFAULT_MOTION_MODEL = CONSTANT_VELOCITY; //NO_MM, CONSTANT_VELOCITY
    const int DEFAULT_MESTIMATOR = STUDENT;
    const int DEFAULT_FINEST_LEVEL = 0; //0->640x480, 1->320x240, 2-> 160x120, ...
    const int DEFAULT_SIGMA = SIGMA_PDF;
    const int DEFAULT_WEIGHTING = INDEPENDENT;
    const int DEFAULT_WARPING = WARP_FIRST;
    const int DEFAULT_ODO_KF_COUNT = 9999999;
    const int DEFAULT_INTEGR_KF_COUNT = 9999999;
    const float DEFAULT_VISRATIO_ODO = 0.9;
    const float DEFAULT_VISRATIO_INTEGR = 0.7;
    const int DEFAULT_TERMINATION = ALL_ITERS;
    const int DEFAULT_IMAGE_FILTERING = NO_FILTERS;
    const int DEFAULT_NSAMPLES = 10000;
    

    /** \brief Camera intrinsics structure
    */ 
    struct Intr
    {
      float fx, fy, cx, cy, k1, k2, k3, k4, k5;
      Intr () {}
      Intr (float fx_, float fy_, float cx_, float cy_, 
             float k1_=0.f, float k2_=0.f, float k3_=0.f, float k4_=0.f, float k5_=0.f) : 
          fx (fx_), fy (fy_), cx (cx_), cy (cy_), 
          k1(k1_), k2(k2_), k3(k3_), k4(k4_), k5(k5_) {}

      Intr operator () (int level_index) const
      { 
        int div = 1 << level_index; 
        return (Intr (fx / div, fy / div, cx / div, cy / div, k1, k2, k3, k4, k5));
      }

      friend inline std::ostream&
      operator << (std::ostream& os, const Intr& intr)
      {
        os << "([f = " << intr.fx << ", " << intr.fy << "] [cp = " << intr.cx << ", " << intr.cy << "])";
        return (os);
      }
    };
    
    struct DepthDist
    {
      float c1, c0, alpha0, alpha1, alpha2, kd1, kd2, kd3, kd4, kd5, kd6, kd7, kd8;
      int xshift, yshift;
      DepthDist () {}
      DepthDist (float c1_, float c0_, float alpha0_ = 1.f, float alpha1_=0.f , float alpha2_ = 0.f,
                  float kd1_=0.f, float kd2_=0.f, float kd3_=0.f, float kd4_=0.f, 
                  float kd5_=0.f, float kd6_=0.f, float kd7_=0.f, float kd8_=0.f,
                  int xshift_ = 4, int yshift_ = 4) : 
                  c1(c1_), c0(c0_), alpha0(alpha0_), alpha1(alpha1_), alpha2(alpha2_),
                  kd1(kd1_), kd2(kd2_), kd3(kd3_), kd4(kd4_), kd5(kd5_), kd6(kd6_), kd7(kd7_), kd8(kd8_), 
                  xshift(xshift_), yshift(yshift_) {}
                  
    };

    
    /** \brief 3x3 Matrix for device code
    */ 
    struct Mat33
    {
      float3 data[3];
    };

    /** \brief Light source collection
    */ 
    struct LightSource
    {
      float3 pos[1];
      int number;
    };

    //////////////////
    //Debug utils
    void showGPUMemoryUsage();
    

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //   Images (Daniel)
    
    float 
    loadDepthCalibration (DepthDist dp, int rows, int cols, int numSMs = -1);

    /** \brief Computes depth pyramid
      * \param[in] src source
      * \param[out] dst destination
      */      
    float
    pyrDownDepth (DepthMapf& src, DepthMapf& dst, int numSMs = -1);  
    
    /** \brief Computes depth pyramid
      * \param[in] src source
      * \param[out] dst destination
      */  
    float
    pyrDownDepthNN (const DepthMapf& src, DepthMapf& dst, int numSMs = -1);  

    /** \brief Computes intensity pyramid
      * \param[in] src source
      * \param[out] dst destination
      */      
    float
    pyrDownIntensity (IntensityMapf& src, IntensityMapf& dst, int numSMs = -1);  
    
    float
    pyrDownIntensityBilinear (IntensityMapf& src, IntensityMapf& dst, int numSMs = -1);  
    
    float
    pyrDownRGB (const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst, int numSMs = -1);

    void
    convertDepth2Float (const DepthMap& src, DepthMapf& dst);
    
    void
    convertFloat2RGB (const IntensityMapf& src, PtrStepSz<uchar3> dst);

    void
    convertDepth2InvDepth (const DepthMap& src, DepthMapf& dst, float factor_depth);

    /** \brief Computes 8 bits intensity map from 24 bits rgb map
      * \param[in] RGB map source
      * \param[out] intensity map destination
      */              
    void
    computeIntensity (const PtrStepSz<uchar3>& src, IntensityMapf& dst);
    
    void
    decomposeRGBInChannels (const PtrStepSz<uchar3>& src, IntensityMapf& r_dst, IntensityMapf& g_dst, IntensityMapf& b_dst);

    /** \brief Computes intensity gradient
      * \param[in] intensity map 
      * \param[out] horizontal intensity gradient
      * \param[out] vertical intensity gradient
      */             
    float
    computeGradientIntensity (const IntensityMapf& src, GradientMap& dst_hor, GradientMap& dst_vert, int numSMs = -1);  

    /** \brief Computes depth gradient
      * \param[in] depth map 
      * \param[out] horizontal depth gradient
      * \param[out] vertical depth gradient
      */        
    float
    computeGradientDepth (const DepthMapf& src, GradientMap& dst_hor, GradientMap& dst_vert, int numSMs = -1);  
   

    /** \brief Copies intensity and depth maps
      * \param[in] src depth map 
      * \param[in] src intensity map 
      * \param[out] dst depth map 
      * \param[out] dst intensity map 
      */     
    void 
    copyImages (const DepthMapf& src_depth, const IntensityMapf& src_int,
                DepthMapf& dst_depth,  IntensityMapf& dst_int);

    void 
    copyImage (const DeviceArray2D<float>& src, DeviceArray2D<float>& dst);

    void 
    copyImageRGB (const PtrStepSz<uchar3>& src, PtrStepSz<uchar3> dst);
    
    
    void 
    initialiseWeightKeyframe(const DepthMapf& src_depth, DeviceArray2D<float>& dst_weight);
    
    template<typename T>
    void initialiseDeviceMemory2D( DeviceArray2D<T>& src, T val, int numSMs = -1);
    
    void
    generateDepthf (const Mat33& R_inv, const float3& t, const MapArr& vmap, DepthMapf& dst);

    ///Median functions//////////////////////////////////////////////
    float 
    computeErrorGridStride (const DeviceArray2D<float>& im1, const DeviceArray2D<float>& im0, DeviceArray<float>& error, int Nsamples = 9999999, int numSMs = -1);   
    
    
      
    float
    computeChiSquare (DeviceArray<float>& error_int,  DeviceArray<float>& error_depth,  float sigma_int, float sigma_depth, int Mestimator, float &chi_square, float &chi_test, float &Ndof, int numSMs = -1);  
    
    float
    computeSigmaPdf (DeviceArray<float>& error,  float& bias, float& sigma, int Mestimator, int numSMs = -1);
    
    float  
    computeSigmaAndNuStudent (DeviceArray<float>& error, float &bias, float &sigma, float &nu, int Mestimator, int numSMs = -1);
    
    float  
    computeNuStudent (DeviceArray<float>& error, float &bias, float &sigma, float &nu, int numSMs = -1);
      
    ///Visual Odometry functions ////////////////////////////////
                       
    float buildSystemGridStride (const float3 delta_trans, const float3 delta_rot,
                       const DepthMapf& W0, const IntensityMapf& I0,
                       const GradientMap& gradW0_x, const GradientMap& gradW0_y, 
                       const GradientMap& gradI0_x, const GradientMap& gradI0_y,		
                       const DepthMapf& W1, const IntensityMapf& I1,	
                       int Mestimator, int weighting,
                       float sigma_depth, float sigma_int,
                       float bias_depth, float bias_int,
                       const Intr& intr, const int size_A,
                       DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, float_type* matrixA_host, float_type* vectorB_host,
                       int numSMs = -1);  
                       
    float buildSystemStudentNuGridStride (const float3 delta_trans, const float3 delta_rot,
                       const DepthMapf& W0, const IntensityMapf& I0,
                       const GradientMap& gradW0_x, const GradientMap& gradW0_y, 
                       const GradientMap& gradI0_x, const GradientMap& gradI0_y,		
                       const DepthMapf& W1, const IntensityMapf& I1,	
                       int Mestimator, int weighting,
                       float sigma_depth, float sigma_int, 
                       float bias_depth, float bias_int,
                       float nu_depth, float nu_int,
                       const Intr& intr, const int size_A,
                       DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, float_type* matrixA_host, float_type* vectorB_host,
                       int numSMs = -1);  
    
                       
    /** \brief Warps I1 towards I0
      * \param[in] curr intensity map
      * \param[in] warped intensity map
      * \param[in] prev inv depth map
      * \param[in] current R^T = 1_R^0
      * \param[in] current -R^T * t = t_1->0
      * \param[in] camera intrinsics
      */
    float warpIntensityWithTrafo3DInvDepth  (IntensityMapf& src, IntensityMapf& dst, const DepthMapf& depthinv_prev, 
                                             Mat33 inv_rotation, float3 inv_translation, const Intr& intr, int numSMs = -1);     

    /** \brief Warps D1 towards D0
      * \param[in] curr inv depth map
      * \param[in] warped inv depth map
      * \param[in] prev  inv depth map
      * \param[in] current R^T = 1_R^0
      * \param[in] current -R^T * t = t_1->0
      * \param[in] camera intrinsics
      * \param[in] current R = 0_R^1
      */        					   			 
    float 
    warpInvDepthWithTrafo3D  (DepthMapf& src, DepthMapf& dst, const DepthMapf& depth_prev, 
                                    Mat33 inv_rotation, float3 inv_translation, const Intr& intr, int numSMs = -1);  
                                    
    float 
    warpInvDepthWithTrafo3DWeighted (DepthMapf& src, DepthMapf& dst, const DepthMapf& depth_prev, DeviceArray2D<float>& weight_warped,
                             Mat33 inv_rotation_proj, float3 inv_translation_proj, const Intr& intr, int numSMs = -1) ;
                           	
    float 
    registerDepthinv(const DepthMapf& src, DepthMapf& intermediate, DeviceArray2D<int>& intermediate_as_int, DepthMapf& dst, 
                      const Mat33 dRc_proj, float3 t_dc_proj, const Mat33 cRd_proj, int numSMs = -1);
                                     
    float 
    integrateWarpedFrame(const DepthMapf& warped_depth_src, const DeviceArray2D<float>& warped_weight_src,
                                DepthMapf& depth_dst, DeviceArray2D<float>& weight_dst, int numSMs = -1);
                                
    float 
    integrateWarpedRGB (const DepthMapf& depth_warped_src, 
                        const IntensityMapf& r_warped_src, const IntensityMapf& g_warped_src, const IntensityMapf& b_warped_src,
                        const DeviceArray2D<float>& weight_warped_src,
                        DepthMapf& depth_dst, PtrStepSz<uchar3> colors_dst, DeviceArray2D<float>& weight_dst, int numSMs = -1);
                                
    float 
    initialiseWarpedWeights (DeviceArray2D<float>& warped_weights, int numSMs = -1);
                                
    float 
    getVisibilityRatioWithOverlapMask (const DepthMapf& depth_src, const DepthMapf& depth_dst,
                        Mat33 rotation, float3 translation, const Intr& intr, float& visibility_ratio, float geom_tol,
                        BinaryMap& overlap_mask, int numSMs = -1);
                        
    float 
    getVisibilityRatio (const DepthMapf& depth_src, const DepthMapf& depth_dst,
                        Mat33 rotation, float3 translation, const Intr& intr, float& visibility_ratio, float geom_tol, int numSMs = -1);
                        
    
                                
    void
    createVMap (const Intr& intr, const DepthMapf& depth, MapArr& vmap, int numSMs = -1);
    
    void
    createNMap (const MapArr& vmap, MapArr& nmap, int numSMs = -1);
    
    void
    createNMapGradients (const Intr& intr, const DepthMapf& depth_inv, const GradientMap& grad_x, const GradientMap& grad_y, MapArr& nmap, int numSMs = -1);
    
    void
    createNMapEigen (const MapArr& vmap, MapArr& nmap);
    
    void
    transformMaps (const MapArr& vmap_src, const MapArr& nmap_src,
                            const Mat33& Rmat, const float3& tvec,
                            MapArr& vmap_dst, MapArr& nmap_dst);
                            
    /** \brief Performs resize of vertex map to next pyramid level by averaging each four points
      * \param[in] input vertext map
      * \param[out] output resized vertex map
      */
    void 
    resizeVMap (const MapArr& input, MapArr& output);
    
    /** \brief Performs resize of vertex map to next pyramid level by averaging each four normals
      * \param[in] input normal map
      * \param[out] output vertex map
      */
    void 
    resizeNMap (const MapArr& input, MapArr& output);
    
    struct float8  { float x, y, z, w, c1, c2, c3, c4; };
    struct float12 { float x, y, z, w, normal_x, normal_y, normal_z, n4, c1, c2, c3, c4; };
    
    void
    generateImage (const MapArr& vmap, const MapArr& nmap, const LightSource& light, 
                              PtrStepSz<uchar3> dst);
                              
                              void
    generateImageRGB (const MapArr& vmap, const MapArr& nmap, 
                      const PtrStepSz<uchar3>& rgb,
                      const LightSource& light, PtrStepSz<uchar3> dst);
                                
    void 
    paint3DView(const PtrStep<uchar3>& colors, PtrStepSz<uchar3> dst, float colors_weight);
    
    void
    generateDepth (const Mat33& R_inv, const float3& t, const MapArr& vmap, DepthMap& dst);
  
    void
    generateDepthf (const Mat33& R_inv, const float3& t, const MapArr& vmap, DepthMapf& dst);
    
    float
    bilateralFilter (const DeviceArray2D<float>& src, DeviceArray2D<float>& dst, const float sigma_floatmap, int numSMs = -1);
    
    float 
    undistortIntensity(IntensityMapf& src, IntensityMapf& dst, const Intr& intr_int, int numSMs = -1);    
    
    float 
    undistortDepthInv(const DepthMapf& src,  DepthMapf& src_corr, DepthMapf& dst, const Intr& intr_depth, const DepthDist& dp, int numSMs = -1);    
    
    //Cuda test functions
    float 
    cudaCreateTexture (DeviceArray2D<float>& dummy, DepthDist dp, int rows, int cols, int numSMs = -1);    
    
    float 
    cudaFetchFromTexture (DeviceArray2D<float>& dummy, DepthDist dp, int cols, int rows, int numSMs = -1);    
    
    float 
    cudaComputeOperations (DeviceArray2D<float>& dummy, DepthDist dp, int rows, int cols, int numSMs = -1);    
    
    float     
    cudaReleaseTexture ();

    /** \brief synchronizes CUDA execution */
    inline void 
    sync () { cudaSafeCall (cudaStreamSynchronize(0)); }
    
    template<class D, class Matx> D&
    device_cast (Matx& matx)
    {
      return (*reinterpret_cast<D*>(matx.data ()));
    }   
    
  }
}


#endif /* VISODO_INTERNAL_HPP_RGBD_ */
