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

/**
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
http://stackoverflow.com/questions/7961792/device-constant-const
In CUDA __constant__is a variable type qualifier that indicates the variable being declared is to be stored in device constant memory. Quoting section B 2.2 of the CUDA programming guide

    The __constant__ qualifier, optionally used together with __device__, declares a variable that:

        Resides in constant memory space,
        Has the lifetime of an application,
        Is accessible from all the threads within the grid and from the host through the runtime library (cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol() for the runtime API and cuModuleGetGlobal() for the driver API).

In CUDA, constant memory is a dedicated, static, global memory area accessed via a cache (there are a dedicated set of PTX load instructions for its purpose) which are uniform and read-only for all threads in a running kernel. But the contents of constant memory can be modified at runtime through the use of the host side APIs quoted above. 
*/

#include "device.hpp"
#include <stdio.h>


namespace RGBID_SLAM
{
  namespace device
  {
  
    __constant__ float MAX_DEPTH = 15.f;  
    __constant__ float DEPTHINV_INTEGR_TH = 0.0075f;
    
    
    namespace reg
    {
      enum
      {
        BLOCK_X = 32,
        BLOCK_Y = 2
      };
    }
    
    
    namespace warp
    {
      enum
      {
        CTA_SIZE_X = 32,
        #if MIN_OPTIMAL_THREADS_PER_BLOCK > 32
          CTA_SIZE_Y = MIN_OPTIMAL_THREADS_PER_BLOCK / CTA_SIZE_X,
        #else
          CTA_SIZE_Y = 1,
        #endif
        CTA_SIZE = CTA_SIZE_X*CTA_SIZE_Y
      };
    }
    
    template<int CTA_SIZE_, typename T>
    static __device__ __forceinline__ void reduce(volatile T* buffer)
    {
      int tid = Block::flattenedThreadId();
      T val =  buffer[tid];

      if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = val + buffer[tid + 512]; __syncthreads(); }
      if (CTA_SIZE_ >=  512) { if (tid < 256) buffer[tid] = val = val + buffer[tid + 256]; __syncthreads(); }
      if (CTA_SIZE_ >=  256) { if (tid < 128) buffer[tid] = val = val + buffer[tid + 128]; __syncthreads(); }
      if (CTA_SIZE_ >=  128) { if (tid <  64) buffer[tid] = val = val + buffer[tid +  64]; __syncthreads(); }

      if (tid < 32)
      {
        if (CTA_SIZE_ >=   64) { buffer[tid] = val = val + buffer[tid +  32]; }
        if (CTA_SIZE_ >=   32) { buffer[tid] = val = val + buffer[tid +  16]; }
        if (CTA_SIZE_ >=   16) { buffer[tid] = val = val + buffer[tid +   8]; }
        if (CTA_SIZE_ >=    8) { buffer[tid] = val = val + buffer[tid +   4]; }
        if (CTA_SIZE_ >=    4) { buffer[tid] = val = val + buffer[tid +   2]; }
        if (CTA_SIZE_ >=    2) { buffer[tid] = val = val + buffer[tid +   1]; }
      }
    }  
    
    __device__ __forceinline__ float
    registerPixel(float& xc, float& yc, int xd, int yd, float wd, const Mat33& cRd_proj, const float3& t_cd_proj)
    {
      float zd = 1.f/wd;
      float3 Xd;
      
      Xd.x = __int2float_rn(xd)*zd;
      Xd.y = __int2float_rn(yd)*zd;
      Xd.z = zd;
      
      float3 Xc = cRd_proj*Xd + t_cd_proj;
      
      float wc = 1.f / Xc.z; 
      xc = Xc.x * wc;
      yc = Xc.y *wc;    
      
      return wc;  
    }
    
    __device__ __forceinline__ float
    registerPixelTranslationOnly(float& xc, float& yc, int xd, int yd, float wd, const float3& t_dc_proj)
    {
      float zd = 1.f/wd;
      float3 Xd;
      
      Xd.x = __int2float_rn(xd)*zd;
      Xd.y = __int2float_rn(yd)*zd;
      Xd.z = zd;
      
      float3 Xc = Xd - t_dc_proj;
      
      float wc = 1.f / Xc.z; 
      xc = Xc.x * wc;
      yc = Xc.y *wc;    
      
      return wc;  
    }
    
    
    __global__ void
    initialiseRegistrationKernel(PtrStepSz<float> dst, PtrStepSz<int> dst_as_int)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      //Initialise registered image
      for (unsigned int xc = tid_x;   xc < dst.cols; xc += blockDim.x * gridDim.x)
      { 
        for (unsigned int yc = tid_y;   yc < dst.rows; yc += blockDim.y * gridDim.y)    
        {
          dst_as_int.ptr(yc)[xc] = 0;
          dst.ptr(yc)[xc] = numeric_limits<float>::quiet_NaN (); 
        }
      }
    }
    
    
    __global__ void
    conversionRegistrationKernel(PtrStepSz<float> dst, PtrStepSz<int> dst_as_int)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      //Initialise registered image
      for (unsigned int xc = tid_x;   xc < dst.cols; xc += blockDim.x * gridDim.x)
      { 
        for (unsigned int yc = tid_y;   yc < dst.rows; yc += blockDim.y * gridDim.y)    
        {
          if (dst_as_int.ptr(yc)[xc] != 0)
          {
            float val = __int_as_float(dst_as_int.ptr(yc)[xc]);
            dst.ptr(yc)[xc] = val;
          }
        }
      }
    }
    
    __global__ void
    depthinvRegistrationTranslationKernel(PtrStepSz<float> dst, PtrStepSz<int> dst_as_int, const PtrStepSz<float> src, const float3 t_dc_proj, int offset_x, int offset_y)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      //First round: we apply registration only on nn to float pixel
      for (unsigned int xd = tid_x;   xd < src.cols; xd += blockDim.x * gridDim.x)
      { 
        for (unsigned int yd = tid_y;   yd < src.rows; yd += blockDim.y * gridDim.y)    
        {
          float wd = src.ptr(yd)[xd];
                    
          if ((!isnan(wd)))
          {
            float xc;
            float yc;
            
            float w_inter = registerPixelTranslationOnly(xc, yc, xd, yd, wd, t_dc_proj); 
            
            if (w_inter > 0.01f)
            {    
              int int_w_inter = __float_as_int(w_inter);
              
              int x = __float2int_rn(xc) + offset_x;
              int y = __float2int_rn(yc) + offset_y;
             
              atomicMax(&dst_as_int.ptr(y)[x], int_w_inter);
            }
          } 
        }
      }
    }
      
      
      __global__ void
    depthinvRegistrationTranslationWithDilationKernel(PtrStepSz<float> dst, PtrStepSz<int> dst_as_int, const PtrStepSz<float> src, const float3 t_dc_proj, int offset_x, int offset_y)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      //To fill possible holes, due to effects like zooming up, apply a dilation
      for (unsigned int xd = tid_x;   xd < src.cols; xd += blockDim.x * gridDim.x)
      { 
        for (unsigned int yd = tid_y;   yd < src.rows; yd += blockDim.y * gridDim.y)    
        {
          float wd = src.ptr(yd)[xd];
          float w_trans = numeric_limits<float>::quiet_NaN (); 
                    
          if ((!isnan(wd)))
          {
            float xc;
            float yc;
            
            float w_inter = registerPixelTranslationOnly(xc, yc, xd, yd, wd, t_dc_proj); 
            
            if (w_inter > 0.01f)
            {            
              float dilation = w_inter / wd;
              //dilation = 1.f;
              
              int int_w_inter = __float_as_int(w_inter);
              
              int xmin = __float2int_rn(xc - 0.5f*dilation) + offset_x;
              int xmax = __float2int_rn(xc + 0.5f*dilation) + offset_x;
              int ymin = __float2int_rn(yc - 0.5f*dilation) + offset_y;
              int ymax = __float2int_rn(yc + 0.5f*dilation) + offset_y;
              
              for (int x = max(0,xmin); x < min(xmax+1,dst.cols); x++)
              {
                for (int y = max(0,ymin); y < min(ymax+1,dst.rows); y++)
                {
                  if (isnan(dst.ptr(y)[x]))
                  {
                    atomicMax(&dst_as_int.ptr(y)[x], int_w_inter);
                  }
                }
              }
            }
          } 
        }
      }
    }
    
    
     
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////      
   ///////////////////////////////////////////////////////////////////////////////////////////////////////  
    
    __global__ void 
    partialVisibilityWithOverlapMaskKernel (int cols, int rows,  const PtrStepSz<float> depthinv_src, const PtrStep<float> depthinv_dst, PtrStep<float> gbuf,
                                   const Mat33 rotation_proj, const float3 translation_proj, float geom_tol, PtrStepSz<unsigned char> overlap_mask)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
      
      float sum_is_visible = 0.f;
      float sum_is_valid = 0.f;         
       
      for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
        {    
          float w = depthinv_src.ptr (y)[x];  
          
          if (!isnan(w))
          {       
            float x_dst, y_dst, w_dst;
            
            w_dst = registerPixel(x_dst, y_dst, x, y, w, rotation_proj, translation_proj);
            
            unsigned char overlap_flag = 0;
            sum_is_valid += 1.f;   

            if ((x_dst > 0) && (x_dst < (cols-1)) && (y_dst > 0) && (y_dst < (rows-1)))
            {
              //Also consider a point not visible if it is projected in front of corresp. point in dst
              //that would be a point behind the camera or a sign of poor odometry estimation
                
              int x_int = __float2int_rn(x_dst);
              int y_int = __float2int_rn(y_dst);               
              
              
              if (abs(w_dst - depthinv_dst.ptr (y_int)[x_int]) < 0.020f)       
              {         
                sum_is_visible += 1.f;
                overlap_flag = 1;
              }                    
            }
            
            overlap_mask.ptr(y)[x] = overlap_flag;
            
          } 
        }
      }

      __syncthreads ();
      __shared__ float smem_visible[256];
      __shared__ float smem_valid[256];
      int tid = Block::flattenedThreadId ();
      
      __syncthreads ();
      smem_visible[tid] = sum_is_visible;
      smem_valid[tid] = sum_is_valid;
      
      __syncthreads ();
      reduce<256>(smem_visible);
      __syncthreads ();
      reduce<256>(smem_valid);
      __syncthreads ();
      
      if (tid == 0)
      {
        gbuf.ptr(0)[blockIdx.x + gridDim.x * blockIdx.y] = smem_visible[0];
        gbuf.ptr(1)[blockIdx.x + gridDim.x * blockIdx.y] = smem_valid[0];
      }

      return;
    }
      
      
    __global__ void 
    partialVisibilityKernel (int cols, int rows,  const PtrStepSz<float> depthinv_src, const PtrStep<float> depthinv_dst, PtrStep<float> gbuf,
                                   const Mat33 rotation_proj, const float3 translation_proj, float geom_tol)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
      
      float sum_is_visible = 0.f;
      float sum_is_valid = 0.f;
       
       
      for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
        { 
          float w = depthinv_src.ptr (y)[x];  
          
          if (!isnan(w))
          {       
            float x_dst, y_dst, w_dst;
            
            w_dst = registerPixel(x_dst, y_dst, x, y, w, rotation_proj, translation_proj);
            
            sum_is_valid += 1.f;   

            if ((x_dst > 0) && (x_dst < (cols-1)) && (y_dst > 0) && (y_dst < (rows-1)))
            {
              //Also consider a point not visible if it is projected in front of corresp. point in dst
              //that would be a point behind the camera or a sign of poor odometry estimation
                
              int x_int = __float2int_rn(x_dst);
              int y_int = __float2int_rn(y_dst);               
              
              
              if (abs(w_dst - depthinv_dst.ptr (y_int)[x_int]) < 0.020f)       
              {         
                sum_is_visible += 1.f;
              }                    
            }
            
          } 
        }
      }        

      __syncthreads ();
      __shared__ float smem_visible[256];
      __shared__ float smem_valid[256];
      int tid = Block::flattenedThreadId ();
      
      __syncthreads ();
      smem_visible[tid] = sum_is_visible;
      smem_valid[tid] = sum_is_valid;
      
      __syncthreads ();
      reduce<256>(smem_visible);
      __syncthreads ();
      reduce<256>(smem_valid);
      __syncthreads ();
      
      if (tid == 0)
      {
        gbuf.ptr(0)[blockIdx.x + gridDim.x * blockIdx.y] = smem_visible[0];
        gbuf.ptr(1)[blockIdx.x + gridDim.x * blockIdx.y] = smem_valid[0];
      }

      return;
    }
      
    __global__ void
    finalVisibilityReductionKernel(int length, const PtrStep<float> gbuf, float* output)
    {
      const float *beg = gbuf.ptr (blockIdx.x);  //1 block per element in A and b
      const float *end = beg + length;   //length = num_constraints

      int tid = threadIdx.x;
      
      float sum = 0.f;
      for (const float *t = beg + tid; t < end; t += 512)  //Each thread sums #(num_contraints/CTA_SIZE) elements
        sum += *t;

      __syncthreads ();
      __shared__ float smem[512];

      smem[tid] = sum;
      __syncthreads ();

      reduce<512>(smem);

      if (tid == 0)
        output[blockIdx.x] = smem[0];
    }
        
        ///////////////////////////////////////////////////////////

    __global__ void
    trafo3DKernelIntensityWithInvDepthGridStride  (int cols, int rows, PtrStepSz<float> dst, const PtrStepSz<float> depthinv_prev, 
                                                  const Mat33 inv_rotation_proj, const float3 inv_translation_proj, cudaTextureObject_t textureIntensity)
    {
      
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
       
       
      for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
        { 
          float res = numeric_limits<float>::quiet_NaN (); 
          float w = depthinv_prev.ptr (y)[x]; 
            
          if (!isnan(w))
          {       
            float x_src, y_src;            
            registerPixel(x_src, y_src, x, y, w, inv_rotation_proj, inv_translation_proj);
            
            x_src += 0.5f ;
            y_src += 0.5f;
            
            if (  !( __float2int_rd(x_src) < 0 || __float2int_rd(y_src) < 0 
                || __float2int_rd(x_src) >= cols || __float2int_rd(y_src) >= rows ))
            {
              res = tex2D<float>(textureIntensity, x_src , y_src);
              res = max (0.f, min (res, 255.f));
            }
          }
          
          dst.ptr (y)[x] = res;
        }
      }
    }  
		
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    trafo3DKernelInvDepthGridStride  (int cols, int rows, PtrStepSz<float> dst, const PtrStepSz<float> depthinv_prev, 
                                      const Mat33 inv_rotation_proj, const float3 inv_translation_proj, cudaTextureObject_t textureDepthinv)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
       
      for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
        { 
          dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN (); 
          float w = depthinv_prev.ptr (y)[x]; 
            
          if (!isnan(w))
          {       
            float x_src, y_src, w_src3D;            
            w_src3D = registerPixel(x_src, y_src, x, y, w, inv_rotation_proj, inv_translation_proj);
            
            x_src += 0.5f ;
            y_src += 0.5f;
            
            if (  !( __float2int_rd(x_src) < 0 || __float2int_rd(y_src) < 0 
                || __float2int_rd(x_src) >= cols || __float2int_rd(y_src) >= rows ) )
            {
              float w_src2D = tex2D<float>(textureDepthinv, x_src , y_src);
              
              float tz = inv_translation_proj.z;
              float v1_z = (1.f/w_src3D - tz)*w;
              float res = (v1_z / (1.f-w_src2D*tz))*w_src2D;
              
              if (res > 0.f)
                dst.ptr (y)[x] = res;
            }
          }
          
          
          
        }
      }			
    }
    
    
    __global__ void
    trafo3DKernelInvDepthWeightedGridStride  (int cols, int rows, PtrStepSz<float> dst, const PtrStepSz<float> depthinv_prev, PtrStepSz<float> weight_warped,
                                      const Mat33 inv_rotation_proj, const float3 inv_translation_proj, cudaTextureObject_t textureDepthinv)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
       
      for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
        { 
          dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN (); 
          float w = depthinv_prev.ptr (y)[x]; 
            
          if (!isnan(w))
          {       
            float x_src, y_src, w_src3D;            
            w_src3D = registerPixel(x_src, y_src, x, y, w, inv_rotation_proj, inv_translation_proj);
            
            x_src += 0.5f ;
            y_src += 0.5f;
            
            if (  !( __float2int_rd(x_src) < 0 || __float2int_rd(y_src) < 0 
                || __float2int_rd(x_src) >= cols || __float2int_rd(y_src) >= rows ) )
            {
              float w_src2D = tex2D<float>(textureDepthinv, x_src , y_src);
              
              float tz = inv_translation_proj.z;
              float v1_z = (1.f/w_src3D - tz)*w; //=e_z^T(B_Rp^A)p
              float w_factor = 1.f-w_src2D*tz;
              float w_factor2 = w_factor*w_factor;
              float weight_res = (w_factor2*w_factor2) / (v1_z*v1_z);
              float res = (v1_z / w_factor)*w_src2D;
              
              if (res > 0.f)
                dst.ptr (y)[x] = res;
                
              if (weight_res > 0.f)
                weight_warped.ptr(y)[x] = weight_res;
            }
          }
          
        }
      }			
    }
       
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    homographyKernelInvDepthGridStride  (int cols, int rows, PtrStepSz<float> dst, const Mat33 srcHdst, const Mat33 dstHsrc, cudaTextureObject_t textureDepthinv, float offset_x, float offset_y)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
       
      for (unsigned int x = tid_x;   x < dst.cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < dst.rows; y += blockDim.y * gridDim.y)    
        { 
          dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();             
                
          float3 p_dst;
          float3 p_src;
          p_dst.x = __int2float_rn(x);
          p_dst.y = __int2float_rn(y);
          p_dst.z = 1.f;
          
          p_src = srcHdst*p_dst;
          p_src *= (1.f/p_src.z);
          
          float x_src = p_src.x + 0.5f + offset_x;
          float y_src = p_src.y + 0.5f + offset_y;
          
          if (  !( __float2int_rd(x_src) < 0 || __float2int_rd(y_src) < 0 
              || __float2int_rd(x_src) >= cols || __float2int_rd(y_src) >= rows ) )
          {
            float w_src = tex2D<float>(textureDepthinv, x_src , y_src);
            p_dst = dstHsrc*p_src;
            
            float res = w_src / p_dst.z;
            
            if (res > 0.f)
              dst.ptr (y)[x] = res;
          }
        }
      }			
    }
    
    __global__ void 
    integrateWarpedFrameKernel(int cols, int rows,
		    												 const PtrStepSz<float> depthinv_warped_src, const PtrStepSz<float> weight_warped_src,
		    												 PtrStepSz<float> depthinv_dst, PtrStepSz<float> weight_dst)
	  {		  
    	unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
       
      for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
        {         
	        if (!isnan(depthinv_warped_src.ptr(y)[x]))
	        {
	          float w_sum = depthinv_warped_src.ptr(y)[x];
	          float w_KF = depthinv_dst.ptr(y)[x];
	          float dw = fabsf(w_sum-w_KF);
	          
	         	if (isnan(w_KF))// || ((depth_warped_src.ptr(y)[x] - depth_dst.ptr(y)[x]) > 0.02f) ) //no previous measurement or current is occluding (use sigma_invdepth as threshold for oclussion decission)
		       	{
		       		depthinv_dst.ptr(y)[x] = w_sum;
		       		weight_dst.ptr(y)[x] = weight_warped_src.ptr(y)[x];
		       	}
		       	else if ( dw < 3*DEPTHINV_INTEGR_TH )//we dont want to map neither occluding nor occluded pixels from warped frames			   	
		       	{
		       		float new_weight = weight_dst.ptr(y)[x] + weight_warped_src.ptr(y)[x];
				      depthinv_dst.ptr(y)[x] = (w_KF*weight_dst.ptr(y)[x] + w_sum*weight_warped_src.ptr(y)[x]) / new_weight;
				      weight_dst.ptr(y)[x] = new_weight;
				    } //else do nothing (current depth is occluded by KF depth)
			    }
        }
      }
    }
    
      
    __global__ void 
    integrateWarpedRGBKernel(int cols, int rows,
	    												 const PtrStepSz<float> depth_warped_src, 
	    												 const PtrStepSz<float> r_warped_src, const PtrStepSz<float> g_warped_src, const PtrStepSz<float> b_warped_src, 
	    												 const PtrStepSz<float> weight_warped_src,
	    												 PtrStepSz<float> depth_dst, PtrStepSz<uchar3> colors_dst, PtrStepSz<float> weight_dst)
	  {		  
    	unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
       
      for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
        {            
	        if (!isnan(depth_warped_src.ptr(y)[x]) && !isnan(r_warped_src.ptr(y)[x]) && !isnan(g_warped_src.ptr(y)[x]) && !isnan(b_warped_src.ptr(y)[x]))
	        {
	         	if (isnan(depth_dst.ptr(y)[x]))// || ((depth_warped_src.ptr(y)[x] - depth_dst.ptr(y)[x]) > 0.02f) ) //no previous measurement or current is occluding (use sigma_invdepth as threshold for oclussion decission)
		       	{
		       		depth_dst.ptr(y)[x] = depth_warped_src.ptr(y)[x];
		       		(colors_dst.ptr(y)[x]).x = __float2int_rn(r_warped_src.ptr(y)[x]);
		       		(colors_dst.ptr(y)[x]).y = __float2int_rn(g_warped_src.ptr(y)[x]);
		       		(colors_dst.ptr(y)[x]).z = __float2int_rn(b_warped_src.ptr(y)[x]);
		       		weight_dst.ptr(y)[x] = weight_warped_src.ptr(y)[x];
		       	}
		       	else if ( ((depth_dst.ptr(y)[x] - depth_warped_src.ptr(y)[x] ) < DEPTHINV_INTEGR_TH) && ((depth_warped_src.ptr(y)[x] - depth_dst.ptr(y)[x]) < DEPTHINV_INTEGR_TH) )  //we dont want to map neither occluding nor occluded pixels from warped frames			   	
		       	{
		       		float new_weight = weight_dst.ptr(y)[x] + weight_warped_src.ptr(y)[x];
				      depth_dst.ptr(y)[x] = (depth_dst.ptr(y)[x]*weight_dst.ptr(y)[x] + depth_warped_src.ptr(y)[x]*weight_warped_src.ptr(y)[x]) / new_weight;
				      (colors_dst.ptr(y)[x]).x = __float2int_rn(  (__int2float_rn( (colors_dst.ptr(y)[x]).x ) *weight_dst.ptr(y)[x] + r_warped_src.ptr(y)[x]*weight_warped_src.ptr(y)[x]) / new_weight  );
				      (colors_dst.ptr(y)[x]).y = __float2int_rn(  (__int2float_rn( (colors_dst.ptr(y)[x]).y ) *weight_dst.ptr(y)[x] + g_warped_src.ptr(y)[x]*weight_warped_src.ptr(y)[x]) / new_weight  );
				      (colors_dst.ptr(y)[x]).z = __float2int_rn(  (__int2float_rn( (colors_dst.ptr(y)[x]).z ) *weight_dst.ptr(y)[x] + b_warped_src.ptr(y)[x]*weight_warped_src.ptr(y)[x]) / new_weight  );
				      weight_dst.ptr(y)[x] = new_weight;
				    } //else do nothing (current depth is occluded by KF depth)
			    }
        }
      }
    }
      
      
      
    //////////////////////////////////////////////////////////////
    /////////BRIDGE FUNCTIONS//////////////////////////////
    //////////////////////////////////////////////////////////
    
    
    
    
    
    float registerDepthinv(const DepthMapf& src, DepthMapf& intermediate, DeviceArray2D<int>& intermediate_as_int, DepthMapf& dst, 
                            const Mat33 dRc_proj, float3 t_dc_proj, const Mat33 cRd_proj, int numSMs)
    {
      cudaTimer timer;  
      
      int offset_x = (intermediate.cols() - src.cols()) / 2;
      int offset_y = (intermediate.rows() - src.rows()) / 2;
      //TODO: initialise intermediate
      
      {
        dim3 block (warp::CTA_SIZE_X, warp::CTA_SIZE_Y);
        
        int gridX = divUp (src.cols() , block.x);
        int gridY = divUp (src.rows(), block.y);   
        
        computeBlockDim2D(gridX, gridY, warp::CTA_SIZE_X, warp::CTA_SIZE_Y, src.cols(), src.rows(), numSMs);
        
        dim3 grid (gridX, gridY); 
        
        initialiseRegistrationKernel<<<grid, block>>>(intermediate, intermediate_as_int);
        
        //depthinvRegistrationTranslationKernel<<<grid, block>>>(intermediate, intermediate_as_int, src, t_dc_proj, offset_x, offset_y);
        
        cudaSafeCall (cudaStreamSynchronize(0));
        cudaSafeCall ( cudaGetLastError () );	
        
        depthinvRegistrationTranslationWithDilationKernel<<<grid, block>>>(intermediate, intermediate_as_int, src, t_dc_proj, offset_x, offset_y);
        
        cudaSafeCall (cudaStreamSynchronize(0));
        cudaSafeCall ( cudaGetLastError () );	
        
        
        conversionRegistrationKernel<<<grid, block>>>(intermediate, intermediate_as_int);
        
        cudaSafeCall (cudaStreamSynchronize(0));
        cudaSafeCall ( cudaGetLastError () );	
      }
      
      {      
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = intermediate.ptr();
        resDesc.res.pitch2D.pitchInBytes =  intermediate.step();
        resDesc.res.pitch2D.width = intermediate.cols();
        resDesc.res.pitch2D.height = intermediate.rows();
        resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.pitch2D.desc.x = 32; // bits per channel 
        resDesc.res.pitch2D.desc.y = 0; 
        
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint; //initial intensity maps are NaN free
        texDesc.normalizedCoords = false;
        
        cudaTextureObject_t textureDepthinv;
        cudaCreateTextureObject(&textureDepthinv, &resDesc, &texDesc, NULL);

        
        dim3 block (warp::CTA_SIZE_X, warp::CTA_SIZE_Y);
        
        int gridX = divUp (dst.cols (), block.x);
        int gridY = divUp (dst.rows (), block.y);
        
        computeBlockDim2D(gridX, gridY, warp::CTA_SIZE_X, warp::CTA_SIZE_Y, dst.cols(), dst.rows(), numSMs);
        
        dim3 grid (gridX, gridY);
        
        cudaFuncSetCacheConfig (trafo3DKernelInvDepthGridStride, cudaFuncCachePreferL1);
        homographyKernelInvDepthGridStride<<<grid, block>>>(intermediate.cols(), intermediate.rows(),dst, dRc_proj, cRd_proj, textureDepthinv, (float) offset_x, (float) offset_y);    

        cudaSafeCall (cudaStreamSynchronize(0));
        cudaSafeCall ( cudaGetLastError () );	
        
        cudaSafeCall (cudaDestroyTextureObject(textureDepthinv));
      }
      
      
      
      //{
        //dim3 block (32, 32);
        //dim3 grid (1, 1);
        
        //int stride_x = divUp (dst.cols() , block.x*grid.x);
        //int stride_y = divUp (dst.rows() , block.y*grid.y);
        
        //int zbuffer_iters = 1;
        
        //for (int i=0; i < zbuffer_iters; i++)
        //{      
          //depthinvDilationKernel<<<grid, block>>>(dst, rect_corner_pos, dst_in_src, stride_x, stride_y);
      
          //cudaSafeCall (cudaStreamSynchronize(0));
          //cudaSafeCall ( cudaGetLastError () );
        //}
      //}
      
      return timer.getTime();
    };
    
    
    float 
    getVisibilityRatio (const DepthMapf& depth_src, const DepthMapf& depth_dst, 
                        Mat33 rotation_proj, float3 translation_proj, const Intr& intr, float& visibility_ratio, float geom_tol, int numSMs)  
    {
      cudaTimer timer;      

      dim3 block (32, 8);
      
      int gridX = divUp (depth_src.cols (), block.x);
      int gridY = divUp (depth_src.rows (), block.y);  
      
      computeBlockDim2D(gridX, gridY, 32, 8, depth_src.cols(), depth_src.rows(), numSMs);
      
      dim3 grid (gridX, gridY);
      
      DeviceArray2D<float> gbuf;
      float gbuf_length = grid.x*grid.y;
      gbuf.create(2,gbuf_length);
      
      DeviceArray<float> output_dev;
      output_dev.create(2);
      
      float output_host[2];

      partialVisibilityKernel<<<grid, block>>>(depth_src.cols(), depth_src.rows(),  depth_src, depth_dst, gbuf, rotation_proj, translation_proj, geom_tol);

      cudaSafeCall (cudaStreamSynchronize(0));
      cudaSafeCall ( cudaGetLastError () );
      
      finalVisibilityReductionKernel<<<2,512>>>(gbuf_length, gbuf, output_dev);
      
      cudaSafeCall (cudaStreamSynchronize(0));
      cudaSafeCall ( cudaGetLastError () );
      
      output_dev.download(output_host);
      output_dev.release();
      gbuf.release();
      
      if (output_host[1] < 1.f)
        visibility_ratio = 0.f;
      else
        visibility_ratio = output_host[0] / output_host[1];// / ((float) (depth_src.cols ()*depth_src.rows ()));
      
      return timer.getTime();
    };
    
    
    
    float 
    getVisibilityRatioWithOverlapMask (const DepthMapf& depth_src, const DepthMapf& depth_dst, 
                        Mat33 rotation_proj, float3 translation_proj, const Intr& intr, float& visibility_ratio, float geom_tol,
                        BinaryMap& overlap_mask, int numSMs)  
    {
      cudaTimer timer;      

      dim3 block (32, 8);
      dim3 grid (divUp (depth_src.cols (), block.x), divUp (depth_src.rows (), block.y));
      
      DeviceArray2D<float> gbuf;
      float gbuf_length = grid.x*grid.y;
      gbuf.create(2,gbuf_length);
      
      DeviceArray<float> output_dev;
      output_dev.create(2);
      
      float output_host[2];

      partialVisibilityWithOverlapMaskKernel<<<grid, block>>>(depth_src.cols(), depth_src.rows(),  depth_src, depth_dst, gbuf, rotation_proj, translation_proj, geom_tol, overlap_mask);

      cudaSafeCall (cudaStreamSynchronize(0));
      cudaSafeCall ( cudaGetLastError () );
      
      finalVisibilityReductionKernel<<<2,512>>>(gbuf_length, gbuf, output_dev);
      
      cudaSafeCall (cudaStreamSynchronize(0));
      cudaSafeCall ( cudaGetLastError () );
      
      output_dev.download(output_host);
      
      if (output_host[1] < 1.f)
        visibility_ratio = 0.f;
      else
        visibility_ratio = output_host[0] / output_host[1];// / ((float) (depth_src.cols ()*depth_src.rows ()));
      
      
      return timer.getTime();
    };
    
      
      
      
		  
		  
		float 
    warpIntensityWithTrafo3DInvDepth  (IntensityMapf& src, IntensityMapf& dst, const DepthMapf& depth_prev, 
                                       Mat33 inv_rotation_proj, float3 inv_translation, const Intr& intr, int numSMs)  //rotation and translation trafo -> src_T^dst src=curr, dst="prev"
    {
      cudaTimer timer;      
      
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = src.ptr();
      resDesc.res.pitch2D.pitchInBytes =  src.step();
      resDesc.res.pitch2D.width = src.cols();
      resDesc.res.pitch2D.height = src.rows();
      resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
      resDesc.res.pitch2D.desc.x = 32; // bits per channel 
      resDesc.res.pitch2D.desc.y = 0; 
      
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.readMode = cudaReadModeElementType;
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModeLinear; //initial intensity maps are NaN free
      texDesc.normalizedCoords = false;
      
      cudaTextureObject_t textureIntensity;
      cudaCreateTextureObject(&textureIntensity, &resDesc, &texDesc, NULL);

      
      dim3 block (warp::CTA_SIZE_X, warp::CTA_SIZE_Y);
      
      int gridX = divUp (dst.cols (), block.x);
      int gridY = divUp (dst.rows (), block.y);
      
      computeBlockDim2D(gridX, gridY, warp::CTA_SIZE_X, warp::CTA_SIZE_Y, dst.cols(), dst.rows(), numSMs);
      
      dim3 grid (gridX, gridY);
      cudaFuncSetCacheConfig (trafo3DKernelIntensityWithInvDepthGridStride, cudaFuncCachePreferL1);
      trafo3DKernelIntensityWithInvDepthGridStride<<<grid, block>>>(src.cols(), src.rows(),dst, depth_prev, inv_rotation_proj, inv_translation, textureIntensity);          

      cudaSafeCall (cudaStreamSynchronize(0));
      cudaSafeCall ( cudaGetLastError () );	
      
      cudaSafeCall (cudaDestroyTextureObject(textureIntensity));

      return timer.getTime();
    };
    
    
    ///////////////////////////////////////////////////////////////
    float 
    warpInvDepthWithTrafo3D (DepthMapf& src, DepthMapf& dst, const DepthMapf& depth_prev, 
                             Mat33 inv_rotation_proj, float3 inv_translation_proj, const Intr& intr, int numSMs)  //rotation and translation trafo -> src_T^dst src=curr, dst="prev"
    {
      cudaTimer timer;      

      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = src.ptr();
      resDesc.res.pitch2D.pitchInBytes =  src.step();
      resDesc.res.pitch2D.width = src.cols();
      resDesc.res.pitch2D.height = src.rows();
      resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
      resDesc.res.pitch2D.desc.x = 32; // bits per channel 
      resDesc.res.pitch2D.desc.y = 0; 
      
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.readMode = cudaReadModeElementType;
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModePoint; //initial intensity maps are NaN free
      texDesc.normalizedCoords = false;
      
      cudaTextureObject_t textureDepthinv;
      cudaCreateTextureObject(&textureDepthinv, &resDesc, &texDesc, NULL);

      
      dim3 block (warp::CTA_SIZE_X, warp::CTA_SIZE_Y);
      
      int gridX = divUp (dst.cols (), block.x);
      int gridY = divUp (dst.rows (), block.y);
      
      computeBlockDim2D(gridX, gridY, warp::CTA_SIZE_X, warp::CTA_SIZE_Y, dst.cols(), dst.rows(), numSMs);
      
      dim3 grid (gridX, gridY);
      
      cudaFuncSetCacheConfig (trafo3DKernelInvDepthGridStride, cudaFuncCachePreferL1);
      trafo3DKernelInvDepthGridStride<<<grid, block>>>(src.cols(), src.rows(),dst, depth_prev, inv_rotation_proj, inv_translation_proj, textureDepthinv);    

      cudaSafeCall (cudaStreamSynchronize(0));
      cudaSafeCall ( cudaGetLastError () );	
      
      cudaSafeCall (cudaDestroyTextureObject(textureDepthinv));
      return timer.getTime();

    };  
    
    float 
    warpInvDepthWithTrafo3DWeighted (DepthMapf& src, DepthMapf& dst, const DepthMapf& depth_prev, DeviceArray2D<float>& weight_warped,
                             Mat33 inv_rotation_proj, float3 inv_translation_proj, const Intr& intr, int numSMs)  //rotation and translation trafo -> src_T^dst src=curr, dst="prev"
    {
      cudaTimer timer;      

      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = src.ptr();
      resDesc.res.pitch2D.pitchInBytes =  src.step();
      resDesc.res.pitch2D.width = src.cols();
      resDesc.res.pitch2D.height = src.rows();
      resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
      resDesc.res.pitch2D.desc.x = 32; // bits per channel 
      resDesc.res.pitch2D.desc.y = 0; 
      
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.readMode = cudaReadModeElementType;
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModePoint; //initial intensity maps are NaN free
      texDesc.normalizedCoords = false;
      
      cudaTextureObject_t textureDepthinv;
      cudaCreateTextureObject(&textureDepthinv, &resDesc, &texDesc, NULL);

      
      dim3 block (warp::CTA_SIZE_X, warp::CTA_SIZE_Y);
      
      int gridX = divUp (dst.cols (), block.x);
      int gridY = divUp (dst.rows (), block.y);
      
      computeBlockDim2D(gridX, gridY, warp::CTA_SIZE_X, warp::CTA_SIZE_Y, dst.cols(), dst.rows(), numSMs);
      
      dim3 grid (gridX, gridY);
      
      cudaFuncSetCacheConfig (trafo3DKernelInvDepthGridStride, cudaFuncCachePreferL1);
      trafo3DKernelInvDepthWeightedGridStride<<<grid, block>>>(src.cols(), src.rows(),dst, depth_prev, weight_warped, inv_rotation_proj, inv_translation_proj, textureDepthinv);    

      cudaSafeCall (cudaStreamSynchronize(0));
      cudaSafeCall ( cudaGetLastError () );	
      
      cudaSafeCall (cudaDestroyTextureObject(textureDepthinv));
      return timer.getTime();

    };  
    
    
    float 
 		integrateWarpedFrame (const DepthMapf& depth_warped_src, const DeviceArray2D<float>& weight_warped_src,
                          DepthMapf& depth_dst, DeviceArray2D<float>& weight_dst, int numSMs)  
	  {
	    cudaTimer timer;      

	    dim3 block (warp::CTA_SIZE_X, warp::CTA_SIZE_Y);
	    
	    int gridX = divUp (depth_warped_src.cols (), block.x);
      int gridY = divUp (depth_warped_src.rows (), block.y);  
      
      computeBlockDim2D(gridX, gridY, warp::CTA_SIZE_X, warp::CTA_SIZE_Y, depth_warped_src.cols(), depth_warped_src.rows(), numSMs);
      
      dim3 grid (gridX, gridY);

	    integrateWarpedFrameKernel<<<grid, block>>>(depth_warped_src.cols(), depth_warped_src.rows(),  
	    																						depth_warped_src, weight_warped_src,
	    																						depth_dst, weight_dst);

	    cudaSafeCall (cudaStreamSynchronize(0));
	    cudaSafeCall ( cudaGetLastError () );					

	    return timer.getTime();
	  };	 
    
    
    
    
    
    
		  
	  float 
 		integrateWarpedRGB (const DepthMapf& depth_warped_src, 
 		                    const IntensityMapf& r_warped_src, const IntensityMapf& g_warped_src, const IntensityMapf& b_warped_src,
 		                    const DeviceArray2D<float>& weight_warped_src,
                        DepthMapf& depth_dst, PtrStepSz<uchar3>  colors_dst, DeviceArray2D<float>& weight_dst, int numSMs)  
	  {
	    cudaTimer timer;      

	    dim3 block (warp::CTA_SIZE_X, warp::CTA_SIZE_Y);
	    
	    int gridX = divUp (depth_warped_src.cols (), block.x);
      int gridY = divUp (depth_warped_src.rows (), block.y);  
      
      computeBlockDim2D(gridX, gridY, warp::CTA_SIZE_X, warp::CTA_SIZE_Y, depth_warped_src.cols(), depth_warped_src.rows(), numSMs);
      
      dim3 grid (gridX, gridY);

	    integrateWarpedRGBKernel<<<grid, block>>>(depth_warped_src.cols(), depth_warped_src.rows(), depth_warped_src, 
	                                              r_warped_src, g_warped_src,  b_warped_src,
	                                              weight_warped_src, depth_dst, colors_dst, weight_dst);
	    																						

	    cudaSafeCall (cudaStreamSynchronize(0));
	    cudaSafeCall ( cudaGetLastError () );					

	    return timer.getTime();
	  };	
    
  }
}
