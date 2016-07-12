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
                  
      namespace filter
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
          
          RADIUS = 2,

          DIAMETER = RADIUS * 2 + 1,
          AREA = DIAMETER*DIAMETER
        };
      }

      __constant__ float sigma_space = 5.f;     

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      __global__ void
      bilateralKernel (const PtrStepSz<float> src, 
                      PtrStep<float> dst, 
                      const float sigma_floatmap)
      {
        unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
        unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
         
        for (unsigned int x = tid_x;   x < src.cols; x += blockDim.x * gridDim.x)
        { 
          for (unsigned int y = tid_y;   y < src.rows; y += blockDim.y * gridDim.y)    
          { 
          
            float value = src.ptr (y)[x];
            
            if (isnan(value))
            {
              dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
              continue;
            }

            int tx = min (x + filter::RADIUS + 1, src.cols);
            int ty = min (y + filter::RADIUS + 1, src.rows);

            float sum1 = 0;
            float sum2 = 0;

            for (int cy = max (y - filter::RADIUS, 0); cy < ty; ++cy)
            {
              for (int cx = max (x - filter::RADIUS, 0); cx < tx; ++cx)
              {
                float tmp = src.ptr (cy)[cx];
                
                if (!isnan(tmp))
                {
                  float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                  float floatval_normalised = (value - tmp) / sigma_floatmap;              
                  float sigma2_space_inv_half = 0.5 / (sigma_space*sigma_space);
                  float weight = __expf (-(sigma2_space_inv_half*space2 + 0.5*floatval_normalised*floatval_normalised  ));
                  sum1 += tmp * weight;
                  sum2 += weight;
                }            
              }
            }

            float res =  (sum1 / sum2);
            dst.ptr (y)[x] = res;
          }
        }
      }

      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      float
      bilateralFilter (const DeviceArray2D<float>& src, DeviceArray2D<float>& dst, const float sigma_floatmap, int numSMs)
      {
        cudaTimer timer;      
        
        dim3 block (filter::CTA_SIZE_X, filter::CTA_SIZE_Y);
        
        int gridX = divUp (src.cols (), block.x);
        int gridY = divUp (src.rows (), block.y);   
      
        computeBlockDim2D(gridX, gridY, 32, 2, src.cols(), src.rows(), numSMs);
        
        dim3 grid (gridX, gridY);  
        
        //float inv_sigma_floatmap_half = 0.5f / (sigma_floatmap * sigma_floatmap);

        cudaFuncSetCacheConfig (bilateralKernel, cudaFuncCachePreferL1);
        bilateralKernel<<<grid, block>>>(src, dst, sigma_floatmap);

        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall (cudaStreamSynchronize(0)); 
        
        return timer.getTime();
      };

    
  }
}
      
