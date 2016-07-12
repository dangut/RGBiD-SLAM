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
  
    namespace undist
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
    
    //texture<float, 2, cudaReadModeElementType> texRefDistortion; 
    
    __device__ __forceinline__ void 
    distortPixel(float uu, float vu, float& ud, float& vd, const Intr& intr)
    {
      float r2 = uu*uu+vu*vu;
      float r4 = r2*r2;
      float r6 = r2*r4;
      
      float factor_r = 1.f + intr.k1*r2 + intr.k2*r4 + intr.k5*r6;
      
      ud = factor_r*uu;
      ud+= 2.f*intr.k3*uu*vu +  intr.k4*(r2 + 2.f*uu*uu);
      
      vd = factor_r*vu;
      vd+= 2.f*intr.k4*uu*vu +  intr.k3*(r2 + 2.f*vu*vu);
      
      return;  
    }
    
    __device__ __forceinline__ float
    undistortDepthinvOffset(float u, float v, const DepthDist& dp)
    {
      float r2 = u*u+v*v;
      float r4 = r2*r2;
      float r6 = r2*r4;
      float uv = u*v;
      float u2v = u*u*v;
      float uv2 = u*v*v;
      
      float spatial_offset = dp.kd1*r2 + dp.kd2*r4 + dp.kd3*r6 + dp.kd4*u + dp.kd5*v + dp.kd6*uv + dp.kd7*u2v + dp.kd8*uv2;
      
      return spatial_offset;
    }
    
    __device__ __forceinline__ float
    correctDepthinv(float u, float v, float wd, const DepthDist& dp)
    {
      float spatial_offset = undistortDepthinvOffset(u, v, dp);
      
      float wu = wd + spatial_offset*(dp.alpha0 + dp.alpha1*wd + dp.alpha2*wd*wd);
      wu *= dp.c1;
      wu += dp.c0; 
      
      return wu;
    }
    
    
    __global__ void
    undistortKernel(PtrStepSz<float> dst, const Intr intr, cudaTextureObject_t src_tex)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
       
       
      for (unsigned int xu = tid_x;   xu < dst.cols; xu += blockDim.x * gridDim.x)
      { 
        for (unsigned int yu = tid_y;   yu < dst.rows; yu += blockDim.y * gridDim.y)    
        {
          float res = numeric_limits<float>::quiet_NaN (); 
          
          float uu = (__int2float_rn(xu) - intr.cx) * (1.f/intr.fx);
          float vu = (__int2float_rn(yu) - intr.cy) * (1.f/intr.fy);
          
          float ud;
          float vd;
          
          distortPixel(uu, vu, ud, vd, intr);
          
          float xd = intr.fx*ud + intr.cx + 0.5f;
          float yd = intr.fy*vd + intr.cy + 0.5f;
          
          if (  !( (xd <= 0) || (yd <= 0) || (xd >= dst.cols) || (yd >= dst.rows)) )          
            res = tex2D<float>(src_tex, xd , yd);
            
          dst.ptr(yu)[xu] = res;
        }
      }
      
      return;
    } 
    
    
    __global__ void
    depthinvCorrectionKernel(PtrStepSz<float> dst, const PtrStepSz<float> src, const Intr intr, const DepthDist dp) //invdepth must be in m^-1! beware of calibration!!
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
       
       
      for (unsigned int x = tid_x;   x < dst.cols; x += blockDim.x * gridDim.x)
      { 
        for (unsigned int y = tid_y;   y < dst.rows; y += blockDim.y * gridDim.y)    
        {
          float res = numeric_limits<float>::quiet_NaN (); 
          
          int x_shifted = x-dp.xshift;
          int y_shifted = y-dp.yshift;
          
          if ((x_shifted > 0) &&(y_shifted > 0))
          {
            float u = (__int2float_rn(x) - intr.cx) * (1.f/intr.fx);
            float v = (__int2float_rn(y) - intr.cy) * (1.f/intr.fy);
            
            float val =  src.ptr (y_shifted)[x_shifted];
            
            res = correctDepthinv(u, v, val, dp);
          }
          
          dst.ptr(y)[x] = res;
        }
      }
      return;      
    }
    
    
    float undistortIntensity(IntensityMapf& src, IntensityMapf& dst, const Intr& intr, int numSMs)
    {
      cudaTimer timer;  
      
      ////src image to texture
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
      
      cudaTextureObject_t src_tex;
      cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
      /////////////////////
      
      dim3 block (undist::CTA_SIZE_X, undist::CTA_SIZE_Y);
      
      int gridX = divUp (src.cols() , block.x);
      int gridY = divUp (src.rows(), block.y);   
      
      computeBlockDim2D(gridX, gridY, undist::CTA_SIZE_X, undist::CTA_SIZE_Y, src.cols(), src.rows(), numSMs);
      
      dim3 grid (gridX, gridY);  
      
      undistortKernel<<<grid, block>>>(dst, intr, src_tex);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall(cudaStreamSynchronize(0));
      
      return timer.getTime();
    };
    
    
    float undistortDepthInv(const DepthMapf& src, DepthMapf& src_corr, DepthMapf& dst, const Intr& intr, const DepthDist& dp, int numSMs)
    {
      cudaTimer timer;  
      
      dim3 block (undist::CTA_SIZE_X, undist::CTA_SIZE_Y);
      
      int gridX = divUp (src.cols() , block.x);
      int gridY = divUp (src.rows(), block.y);   
      
      computeBlockDim2D(gridX, gridY, undist::CTA_SIZE_X, undist::CTA_SIZE_Y, src.cols(), src.rows(), numSMs);
      
      dim3 grid (gridX, gridY);  
      
      depthinvCorrectionKernel<<<grid, block>>>(src_corr, src, intr, dp);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall(cudaStreamSynchronize(0));
    
      ////src image to texture
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = src_corr.ptr();
      resDesc.res.pitch2D.pitchInBytes =  src_corr.step();
      resDesc.res.pitch2D.width = src_corr.cols();
      resDesc.res.pitch2D.height = src_corr.rows();
      resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
      resDesc.res.pitch2D.desc.x = 32; // bits per channel 
      resDesc.res.pitch2D.desc.y = 0; 
      
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.readMode = cudaReadModeElementType;
      texDesc.addressMode[0] = cudaAddressModeClamp;
      texDesc.addressMode[1] = cudaAddressModeClamp;
      texDesc.filterMode = cudaFilterModePoint; 
      texDesc.normalizedCoords = false;
      
      cudaTextureObject_t src_tex;
      cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
      /////////////////////
      
      
      
      undistortKernel<<<grid, block>>>(dst, intr, src_tex);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall(cudaStreamSynchronize(0));
      
      return timer.getTime();
    };

  }
}


