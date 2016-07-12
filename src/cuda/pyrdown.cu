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

#include "device.hpp"


namespace RGBID_SLAM
{
  namespace device
  {    
    namespace pyr
    {
      enum
      {
        CTA_SIZE_X = 32,
        #if MIN_OPTIMAL_THREADS_PER_BLOCK > 32
          CTA_SIZE_Y = MIN_OPTIMAL_THREADS_PER_BLOCK / CTA_SIZE_X,
        #else
          CTA_SIZE_Y = 1,
        #endif
        CTA_SIZE = CTA_SIZE_X*CTA_SIZE_Y,
        
        RADIUS_DEPTH = 2,
        RADIUS_INT = 2
      };
    }

    __constant__ float sigma = 1.f;   
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
    pyrDownKernelGridStridef (const PtrStepSz<float> src, PtrStepSz<float> dst, int blur_radius)
    {
      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      for (int x = tid_x;   x < dst.cols; x += blockDim.x * gridDim.x)
      { 
        for (int y = tid_y;   y < dst.rows; y += blockDim.y * gridDim.y)    
        { 
          //float center = src.ptr (2 * y)[2 * x];
          
          float res = numeric_limits<float>::quiet_NaN ();

          int tx = min (2 * x + blur_radius + 1, src.cols);
          int ty = min (2 * y + blur_radius + 1, src.rows);

          float sum1 = 0.f;
          float sum2 = 0.f;
          int count = 0;

          float sigma_space2_inv_half = 0.5f / (sigma *sigma);

          for (int cy = max (0, 2 * y - blur_radius); cy < ty; ++cy)
          {
            for (int cx = max (0, 2 * x - blur_radius); cx < tx; ++cx)
            {
              float val = src.ptr (cy)[cx];
              if (!isnan(val))
              {
                float space2 = (2 * x - cx) * (2 * x - cx) + (2 * y - cy) * (2 * y - cy);
                float weight = __expf (-(space2 * sigma_space2_inv_half));
                sum1 += val*weight;
                sum2 += weight;
                ++count;
              }
            }
          }

          int d = 2*blur_radius + 1;
          int area = d*d;
          if (count > (area / 2))  //if more than half of windowed pixels on lower pyr are OK, we downsample.
            res = sum1 / sum2; 

          dst.ptr (y)[x] = res;
        }
      }
    }
    
    
    __global__ void
    pyrDownKernelRGB (const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst)
    {
      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      for (int x = tid_x;   x < dst.cols; x += blockDim.x * gridDim.x)
      { 
        for (int y = tid_y;   y < dst.rows; y += blockDim.y * gridDim.y)    
        { 
          uchar3 res;
          res.x = 0;
          res.y = 0;
          res.z = 0;

          int tx = min (2 * x + pyr::RADIUS_INT + 1, src.cols);
          int ty = min (2 * y + pyr::RADIUS_INT + 1, src.rows);

          float sum_r = 0.f;
          float sum_g = 0.f;
          float sum_b = 0.f;
          float sum2 = 0.f;
          
          uchar3 val;

          float sigma_space2_inv_half = 0.5f / (sigma *sigma);

          for (int cy = max (0, 2 * y - pyr::RADIUS_INT); cy < ty; ++cy)
          {
            for (int cx = max (0, 2 * x - pyr::RADIUS_INT); cx < tx; ++cx)
            {
              val = src.ptr (cy)[cx];
              unsigned char r = val.x;
              unsigned char g = val.y;
              unsigned char b = val.z;
              
              float space2 = (2 * x - cx) * (2 * x - cx) + (2 * y - cy) * (2 * y - cy);
              float weight = __expf (-(space2 * sigma_space2_inv_half));

              sum_r +=  weight*__int2float_rn(r);
              sum_g +=  weight*__int2float_rn(g);
              sum_b +=  weight*__int2float_rn(b);
              
              sum2 += weight;          
            }
          }

          res.x =  min(__float2int_rn(sum_r/sum2),255);
          res.y = min(__float2int_rn(sum_g/sum2),255);
          res.z =  min(__float2int_rn(sum_b/sum2),255);
          
          dst.ptr (y)[x] = res;
        }
      }
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    float
    pyrDownIntensity (IntensityMapf& src, IntensityMapf& dst, int numSMs)
    {
      cudaTimer timer;   
      
      dst.create (src.rows () / 2, src.cols () / 2);

      dim3 block (pyr::CTA_SIZE_X, pyr::CTA_SIZE_Y);
      
      int gridX = divUp (dst.cols (), block.x);
      int gridY = divUp (dst.rows (), block.y);  
      
      computeBlockDim2D(gridX, gridY, pyr::CTA_SIZE_X, pyr::CTA_SIZE_Y, dst.cols(), dst.rows(), numSMs);
      
      dim3 grid (gridX, gridY);
      
      cudaFuncSetCacheConfig (pyrDownKernelGridStridef, cudaFuncCachePreferL1);
      pyrDownKernelGridStridef<<<grid, block>>>(src, dst, pyr::RADIUS_INT);	 
      
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
      
      return timer.getTime();
    };
    
    float
    pyrDownDepth (DepthMapf& src, DepthMapf& dst, int numSMs)
    {
      cudaTimer timer;   
      
      dst.create (src.rows () / 2, src.cols () / 2);

      dim3 block (pyr::CTA_SIZE_X, pyr::CTA_SIZE_Y);
      
      int gridX = divUp (dst.cols (), block.x);
      int gridY = divUp (dst.rows (), block.y);   
      
      computeBlockDim2D(gridX, gridY, pyr::CTA_SIZE_X, pyr::CTA_SIZE_Y, dst.cols(), dst.rows(), numSMs);
      
      dim3 grid (gridX, gridY);      
      
      cudaFuncSetCacheConfig (pyrDownKernelGridStridef, cudaFuncCachePreferL1);
      pyrDownKernelGridStridef<<<grid, block>>>(src, dst, pyr::RADIUS_DEPTH);
      
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
      
      return timer.getTime();
    };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float
    pyrDownRGB (const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst, int numSMs)
    {
      cudaTimer timer;   

      dim3 block (pyr::CTA_SIZE_X, pyr::CTA_SIZE_Y);
      
      int gridX = divUp (dst.cols , block.x);
      int gridY = divUp (dst.rows, block.y);   
      
      computeBlockDim2D(gridX, gridY, pyr::CTA_SIZE_X, pyr::CTA_SIZE_Y, dst.cols, dst.rows, numSMs);
      
      dim3 grid (gridX, gridY);  

      pyrDownKernelRGB<<<grid, block>>>(src, dst);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
      
      return timer.getTime();
    };
    

  }
}

      
