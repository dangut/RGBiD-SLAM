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
      	 
//*.cu for straightforward misc computations (float depth, float inversedepth, truncateDepth, gradients, RGB2grayscale). 
      	
      	
      	
#include "device.hpp"


namespace RGBID_SLAM
{
  namespace device
  {

    namespace grad
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

    
    /////////////////////////////////////////////////////////////////////
    __global__ void
    depth2floatKernel (const PtrStepSz<ushort> src, PtrStep<float> dst) 
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= src.cols || y >= src.rows)
      return;

      dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();

      int value =  src.ptr (y)[x];

      if (value > 0)
        dst.ptr (y)[x] = (__int2float_rn( max( 0 , min( value , 10000 ) ))) / 1000.f; 
      
      return;
    }
    
    /////////////////////////////////////////////////////////////////////
    __global__ void
    depth2invDepthKernel  (const PtrStepSz<ushort> src, PtrStep<float> dst, float factor_depth) 
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= src.cols || y >= src.rows)
        return;

      dst.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();

      int value =  src.ptr (y)[x];

      if (value > 0)
      {
        dst.ptr (y)[x] = (1.f/factor_depth)*1000.f / __int2float_rn( max( 0 , min( value , 10000 ) )); 
      }
      
      return;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    intensityKernel(const PtrStepSz<uchar3> src, 
    PtrStep<float> dst, int cols, int rows) 
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= cols || y >= rows)
        return;

      uchar3 value = src.ptr (y)[x];
      unsigned char r = value.x;
      unsigned char g = value.y;
      unsigned char b = value.z;

      dst.ptr (y)[x] =  max ( 0.f, min ( (  0.2126f * __int2float_rn (r) + 
                                            0.7152f * __int2float_rn (g) + 
                                            0.0722f * __int2float_rn (b)  ), 255.f ) );
      return;
    }
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    decomposeRGBKernel(const PtrStepSz<uchar3> src, 
                       PtrStep<float> r_dst, PtrStep<float> g_dst, PtrStep<float> b_dst,
                       int cols, int rows) 
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= cols || y >= rows)
        return;

      uchar3 value = src.ptr (y)[x];
      unsigned char r = value.x;
      unsigned char g = value.y;
      unsigned char b = value.z;

      r_dst.ptr (y)[x] =  __int2float_rn(r);
      g_dst.ptr (y)[x] =  __int2float_rn(g);
      b_dst.ptr (y)[x] =  __int2float_rn(b);
      
      return;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    __global__ void
    gradientKernel  (const PtrStepSz<float> src, 
                     PtrStep<float> dst_hor,
                     PtrStep<float> dst_vert) 
    {
      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      for (int x = tid_x;   x < src.cols; x += blockDim.x * gridDim.x)
      { 
        for (int y = tid_y;   y < src.rows; y += blockDim.y * gridDim.y)    
        { 

          dst_hor.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
          dst_vert.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
          float value = src.ptr (y)[x];
          
          float res_hor = 0;
          float res_vert = 0;

          for (int dx=-1; dx<2; dx++)
          {
            for (int dy=-1; dy<2; dy++)
            {
              int cx = min (max (0, x + dx), src.cols - 1);
              int cy = min (max (0, y + dy), src.rows - 1);

              int weight_hor = dx * (2 - dy * dy); 
              int weight_vert = dy * (2 - dx * dx);
              float temp_h = src.ptr (cy)[cx];
              float temp_v = temp_h;
                
              res_hor +=  temp_h*weight_hor;
              res_vert +=  temp_v*weight_vert;
            }
          }
        
          //8 is the sum of abs(weights) of sobel operator-> [grad] = [map_units/px]
          dst_hor.ptr (y)[x] =  (res_hor) / 8.f; 
          dst_vert.ptr (y)[x] = (res_vert) / 8.f;
        }
      }
       
      return;
    }
	
    
     
    /////////////////////////////////////////////////////////////////////
    __global__ void
    copyImagesKernel (const PtrStepSz<float> src_depth, const PtrStepSz<float> src_int,
                      PtrStep<float> dst_depth, PtrStep<float> dst_int)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= src_depth.cols || y >= src_depth.rows)
        return;

      float value_depth = src_depth.ptr (y)[x];
      float value_int = src_int.ptr (y)[x];

      dst_depth.ptr (y)[x] = value_depth;
      dst_int.ptr (y)[x] = value_int;
    }

    ///////////////////////////////////////////////////////////////////////

    __global__ void
    copyImageKernel (const PtrStepSz<float> src, PtrStep<float> dst)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= src.cols || y >= src.rows)
        return;

      float value = src.ptr (y)[x];

      dst.ptr (y)[x] = value;
    }
    
    __global__ void
    copyImageRGBKernel (const PtrStepSz<uchar3> src, PtrStep<uchar3> dst)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= src.cols || y >= src.rows)
        return;

      uchar3 value = src.ptr (y)[x];

      dst.ptr (y)[x] = value;
    }
    
    __global__ void
    initialiseWeightKernel (const PtrStepSz<float> src_depth, PtrStep<float> dst_weight)
    {
    	int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= src_depth.cols || y >= src_depth.rows)
        return;
        
      float weight = 1.f;
      
      if (!isnan(src_depth.ptr (y)[x]))
      	weight = 1.f;
      	
      dst_weight.ptr (y)[x] = weight;
    }
    
    __global__ void
    float2ucharKernel(const PtrStep<float> src, PtrStepSz<uchar3> dst,int cols, int rows) 
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= cols || y >= rows)
        return;

      float value_src = src.ptr (y)[x];
      uchar3 value_dst;
      float min_val = 0.f;
      float max_val = 255.f;
      
      if (isnan(value_src))
      {
      	value_dst.x = 200;
      	value_dst.y = 150;
      	value_dst.z = 150;
      }
      else if (isinf(value_src))
      {
      	value_dst.x = 150;
      	value_dst.y = 150;
      	value_dst.z = 250;
      }
      else
      {
      	unsigned char grey_val = max(0, min( __float2int_rn(255*(value_src - min_val)/(max_val-min_val)), 255));
      	value_dst.x = grey_val;
      	value_dst.y = grey_val;
      	value_dst.z = grey_val;
      }

      dst.ptr (y)[x] =  value_dst;
      return;
    }
    
    template<typename T>
    __global__ 
    void initialiseDeviceMemory2DKernel(PtrStepSz<T> src, T val)
    {
      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      for (int x = tid_x;   x < src.cols; x += blockDim.x * gridDim.x)
      { 
        for (int y = tid_y;   y < src.rows; y += blockDim.y * gridDim.y)    
        { 
          src.ptr(y)[x] = val;
        }
      }
    }
    
    
    

    ///////////////////////////////////////////////////////////////
    
    
    
		  
		  
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    convertDepth2Float (const DepthMap& src, DepthMapf& dst)
    {
      dim3 block (32, 8);
      dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

      depth2floatKernel<<<grid, block>>>(src, dst);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    };
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    convertDepth2InvDepth (const DepthMap& src, DepthMapf& dst, float factor_depth)
    {
      dim3 block (32, 8);
      dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

      depth2invDepthKernel<<<grid, block>>>(src, dst, factor_depth);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    computeIntensity (const PtrStepSz<uchar3>& src, IntensityMapf& dst)
    {
      dim3 block (32, 8);
      dim3 grid (divUp (dst.cols(), block.x), divUp (dst.rows(), block.y));
      
      intensityKernel<<<grid, block>>>(src, dst, dst.cols(), dst.rows());
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    };
    
    void
    decomposeRGBInChannels (const PtrStepSz<uchar3>& src, IntensityMapf& r_dst, IntensityMapf& g_dst, IntensityMapf& b_dst)
    {
      dim3 block (32, 8);
      dim3 grid (divUp (r_dst.cols(), block.x), divUp (r_dst.rows(), block.y));
      
      decomposeRGBKernel<<<grid, block>>>(src, r_dst, g_dst, b_dst, r_dst.cols(), r_dst.rows());
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////   
    float
    computeGradientIntensity (const IntensityMapf& src, GradientMap& dst_hor, GradientMap& dst_vert, int numSMs)  //is it correct casting depthmap or Intmap to float map?
    {
      cudaTimer timer;     
      
      dim3 block (32, 8);
      
      int gridX = divUp (src.cols (), block.x);
      int gridY = divUp (src.rows (), block.y);  
      
      computeBlockDim2D(gridX, gridY, 32, 8, src.cols (), src.rows (), numSMs);
      
      dim3 grid (gridX, gridY);

      gradientKernel<<<grid, block>>>(src, dst_hor, dst_vert);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 

      return timer.getTime();
    };
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////      
    float
    computeGradientDepth (const DepthMapf& src, GradientMap& dst_hor, GradientMap& dst_vert, int numSMs)  //is it correct casting depthmap or Intmap to float map?
    {
      cudaTimer timer;     
      
      dim3 block (32, 8);
      
      int gridX = divUp (src.cols (), block.x);
      int gridY = divUp (src.rows (), block.y);  
      
      computeBlockDim2D(gridX, gridY, 32, 8, src.cols (), src.rows (), numSMs);
      
      dim3 grid (gridX, gridY);

      gradientKernel<<<grid, block>>>(src, dst_hor, dst_vert);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 

      return timer.getTime();
    };
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////      
    void 
    copyImages  (const DepthMapf& src_depth, const IntensityMapf& src_int,
                 DepthMapf& dst_depth,  IntensityMapf& dst_int)
    {
      dim3 block (32, 8);
      dim3 grid (divUp (src_depth.cols (), block.x), divUp (src_depth.rows (), block.y));

      copyImagesKernel<<<grid, block>>>(src_depth, src_int, dst_depth, dst_int);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////       
    void 
    copyImage (const DeviceArray2D<float>& src, DeviceArray2D<float>& dst)
    {
      dim3 block (32, 8);
      dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

      copyImageKernel<<<grid, block>>>(src, dst);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    };
    
    void 
    copyImageRGB (const PtrStepSz<uchar3>& src, PtrStepSz<uchar3> dst)
    {
      dim3 block (32, 8);
      dim3 grid (divUp (src.cols, block.x), divUp (src.rows, block.y));

      copyImageRGBKernel<<<grid, block>>>(src, dst);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    };
    
    void 
    initialiseWeightKeyframe(const DepthMapf& src_depth, DeviceArray2D<float>& dst_weight)
    {
    	dim3 block (32, 8);
      dim3 grid (divUp (src_depth.cols (), block.x), divUp (src_depth.rows (), block.y));

      initialiseWeightKernel<<<grid, block>>>(src_depth, dst_weight);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    }
    
    template<typename T>
    void initialiseDeviceMemory2D( DeviceArray2D<T>& src, T val, int numSMs)
    {
      dim3 block (32, 8);
      
      int gridX = divUp(src.cols(), block.x);
      int gridY = divUp(src.rows(), block.y);
      
      computeBlockDim2D(gridX, gridY, 32, 8, src.cols (), src.rows (), numSMs);
      
      dim3 grid (gridX, gridY);
      
      initialiseDeviceMemory2DKernel<T><<<grid, block>>>(src, val);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0));    
    }
    
    template void initialiseDeviceMemory2D (DeviceArray2D<unsigned char>& src, unsigned char val, int numSMs);
    template void initialiseDeviceMemory2D (DeviceArray2D<unsigned int>& src, unsigned int val, int numSMs);
    template void initialiseDeviceMemory2D (DeviceArray2D<char>& src, char val, int numSMs);
    template void initialiseDeviceMemory2D (DeviceArray2D<int>& src, int val, int numSMs);
    template void initialiseDeviceMemory2D (DeviceArray2D<float>& src, float val, int numSMs);
    
    void
    convertFloat2RGB (const IntensityMapf& src, PtrStepSz<uchar3> dst)
    {
      dim3 block (32, 8);
      dim3 grid (divUp (src.cols(), block.x), divUp (src.rows(), block.y));
      
      float2ucharKernel<<<grid, block>>>(src, dst, src.cols(), src.rows());
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaStreamSynchronize(0)); 
    };    
    
    
    void //for debugging
    showGPUMemoryUsage()
    {
      // show memory usage of GPU
      size_t free_byte ;
      size_t total_byte ;

      cudaSafeCall (cudaMemGetInfo( &free_byte, &total_byte )) ;

      double free_db = (double)free_byte ;
      double total_db = (double)total_byte ;
      double used_db = total_db - free_db ;

      std::cout << "GPU memory usage: used =  " << used_db/1024.0/1024.0 << " MB, free = " << free_db/1024.0/1024.0 << " MB, total = " << total_db/1024.0/1024.0 << std::endl;
    };
    
    
	  
 
  }
}

