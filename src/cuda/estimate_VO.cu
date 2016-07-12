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
#include <stdio.h>


namespace RGBID_SLAM
{
  namespace device
  {
    typedef double float_type;    

    __constant__ float THRESHOLD_HUBER_DEV = 1.345f;
    __constant__ float THRESHOLD_TUKEY_DEV = 4.685f;
    __constant__ float STUDENT_DOF_DEV = 5.f;
    
       
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

    struct constraintsHandler
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

      float3 delta_trans;
      float3 delta_rot;

      PtrStep<float> W0;
      PtrStep<float> I0;
      PtrStep<float> W1;
      PtrStep<float> I1;
      PtrStep<float> N0;
      
      Intr intr;

      PtrStep<float> gradI0_x;
      PtrStep<float> gradI0_y;

      PtrStep<float> gradW0_x;
      PtrStep<float> gradW0_y;

      int cols;
      int rows;
      
      int Mestimator;
      int weighting;
      float sigma_int;
      float sigma_depthinv;
      float bias_int;
      float bias_depthinv;
      float nu_int;
      float nu_depthinv;

      mutable PtrStep<float_type> gbuf;
      

      __device__ __forceinline__ float
      computeWeight (float *error) const
      {
        float weight = 1.f;

        if (Mestimator == HUBER)
        {
          if (fabs(*error) > THRESHOLD_HUBER_DEV)
            weight = (THRESHOLD_HUBER_DEV / fabs(*error));
        }
        else if (Mestimator == TUKEY)
        {
          if (fabs(*error) < THRESHOLD_TUKEY_DEV)
          {
            float aux1 = ( (*error) / THRESHOLD_TUKEY_DEV ) * ( (*error) / THRESHOLD_TUKEY_DEV );
            weight = (1.f - aux1) * (1.f - aux1);
          }
          else
            weight = 0.f;
        }
        else if (Mestimator == STUDENT)
        {   
          weight = (STUDENT_DOF_DEV + 1.f) / (STUDENT_DOF_DEV + (*error)*(*error));
        }

        return weight;
      }
      
      
      __device__ __forceinline__ float
      computeWeightStudent (float *error, float nu) const
      {    
        return (nu + 1.f) / (nu + (*error)*(*error));
      }
      
      __device__ __forceinline__ int
      intensityConstraint (int x, int y, float w0, float *row, float *error) const
      {
        float i0 = I0.ptr(y)[x];
        float i1 = I1.ptr(y)[x];
        float gradx = gradI0_x.ptr (y)[x];
        float grady = gradI0_y.ptr (y)[x];

        if (isnan (w0) || isnan (i0) || isnan (i1) || isnan (gradx) || isnan(grady))
          return 0;	          

        float3 p;
        p.x = (__int2float_rn(x) - intr.cx) / intr.fx ;
        p.y = (__int2float_rn(y) - intr.cy) / intr.fy ;
        p.z = 1.f;

        float3 gradI0_times_KP;
        gradI0_times_KP.x = gradx * intr.fx;
        gradI0_times_KP.y = grady * intr.fy;
        gradI0_times_KP.z = - (gradI0_times_KP.x * p.x + gradI0_times_KP.y * p.y);

        float weight = 1.f / sigma_int;

        float3 row_rot = - cross(gradI0_times_KP,p) ;		

        float3 row_trans = (gradI0_times_KP)*w0;		

        float b = (i1-i0);

        *(float3*)&row[0] = row_trans * weight;
        *(float3*)&row[3] = row_rot * weight;  
        
        *error = -b*weight;
        //*error = - (dot ( row_trans, delta_trans )  +  dot ( row_rot, delta_rot ) + b)*weight ; 
        
        return 1;
      }

      __device__ __forceinline__ int
      invDepthConstraint (int x, int y, float w0, float w1, float *row, float *error, float *n_factor) const
      {
        float gradx = gradW0_x.ptr (y)[x] ;
        float grady = gradW0_y.ptr (y)[x] ;

        if (isnan (w0) || isnan (w1) || isnan (gradx) || isnan(grady))
          return 0;	
          
        float3 p;
        p.x = (__int2float_rn(x) - intr.cx) / intr.fx ;
        p.y = (__int2float_rn(y) - intr.cy) / intr.fy ;
        p.z = 1.f;

        float3 gradW0_times_KP;
        gradW0_times_KP.x = (gradx)*intr.fx;  //[mm]/[px]*([px]/[1])* (1/[mm]) = [1]
        gradW0_times_KP.y = (grady)*intr.fy;
        gradW0_times_KP.z = - (gradW0_times_KP.x * p.x + gradW0_times_KP.y * p.y);
        
        
        float3 n = gradW0_times_KP * (1.f/w0);
        n.z += 1.f;
        n = normalized(n);
        float3 p_u = normalized(p);
        
        *n_factor = fabs(dot(n,p_u));//times(dot(n,p_u))??;
        
        
        float weight = 1.f / sigma_depthinv;
        
        float3 row_trans;
        row_trans = gradW0_times_KP*w0; 
        row_trans.z = row_trans.z + w0*w1;

        gradW0_times_KP.z = gradW0_times_KP.z + w1;
        float3 row_rot;
        row_rot = - cross(gradW0_times_KP, p); //a^T->gradD0, b->p. a^T*mcross(b) = mcross(a) * b = cross(a,b)
        
        float b = (w1 - w0);

        *(float3*)&row[0] = row_trans * weight;
        *(float3*)&row[3] = row_rot * weight;
        
        ////////
        *error = - b*weight; 
        //*error = - (dot ( row_trans, delta_trans )  +  dot ( row_rot, delta_rot )  + b)*weight;
        
        return 1;
      }
      
      
      __device__ __forceinline__ void
      computeSystemGridStride () const
      {        
        float row_int[6];
        float row_depthinv[6];	
        float error_int = 0.f;
        float error_depthinv = 0.f; 
        float weight_int = 0.f;
        float weight_depthinv = 0.f;
        float local_sys[TOTAL_SIZE];

        for (int i=0; i < 6; i++)
        { 
          row_int[i] = 0.f;
          row_depthinv[i] = 0.f;
        }
        
        for (int i=0; i < TOTAL_SIZE; i++)
        { 
          local_sys[i] = 0.f;
        }
        
        unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
        unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
        //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
         
        for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
        { 
          for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
          { 
            float n_factor = 1.f;
            weight_int = 0.f;
            weight_depthinv = 0.f;
            
            float w0 = W0.ptr (y)[x];
            float w1 = W1.ptr (y)[x];
            
            if (invDepthConstraint (x, y, w0, w1, row_depthinv, &error_depthinv, &n_factor))
            {
              float error_depthinv_unbiased = error_depthinv - (bias_depthinv / sigma_depthinv);
              weight_depthinv = computeWeight(&error_depthinv_unbiased)*__int2float_rn(1 - (weighting == PHOT_ONLY));
            }
         
            if (intensityConstraint (x, y, w0, row_int, &error_int))
            {   
              float error_int_unbiased = error_int - (bias_int / sigma_int);
              weight_int = computeWeight(&error_int_unbiased)*__int2float_rn(1 - (weighting == GEOM_ONLY)); 
            }
             
            if (weighting == MIN_WEIGHT)
            {
              weight_int = min(weight_depthinv, weight_int);
            }          
            
            int shift = 0;
            for (int i = 0; i < B_SIZE; ++i)        //rows
            {
              #pragma unroll
              for (int j = i; j < B_SIZE; ++j)          // cols + b
              {   
                local_sys[shift++] += weight_int*(row_int[i] * row_int[j]) + n_factor*weight_depthinv*(row_depthinv[i] * row_depthinv[j]);
              }
              
              local_sys[shift++] += weight_int*(row_int[i] *  (error_int)) + n_factor*weight_depthinv*(row_depthinv[i] * (error_depthinv));
            }
           
          }
        }
        
        __shared__ float smem[CTA_SIZE];       
        unsigned int tid_block = Block::flattenedThreadId ();
        
        for (int i = 0; i < TOTAL_SIZE; ++i)        //rows
        {              
          __syncthreads ();        
          smem[tid_block] = local_sys[i];
          __syncthreads ();

          reduce<CTA_SIZE>(smem);
          
          __syncthreads ();

          if (tid_block == 0)
            gbuf.ptr (i)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
        }        
      } 
      
      
      
      __device__ __forceinline__ void
      computeStudentNuSystemGridStride () const
      {        
        float row_int[6];
        float row_depthinv[6];	
        float error_int = 0.f;
        float error_depthinv = 0.f; 
        float weight_int = 0.f;
        float weight_depthinv = 0.f;
        float local_sys[TOTAL_SIZE];

        for (int i=0; i < 6; i++)
        { 
          row_int[i] = 0.f;
          row_depthinv[i] = 0.f;
        }
        
        for (int i=0; i < TOTAL_SIZE; i++)
        { 
          local_sys[i] = 0.f;
        }
        
        unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
        unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
        //unsigned int tid_global =  blockDim.x * gridDim.x * tid_y + tid_x;
         
        for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)
        { 
          for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
          { 
            float n_factor = 1.f;
            weight_int = 0.f;
            weight_depthinv = 0.f;
            
            float w0 = W0.ptr (y)[x];
            float w1 = W1.ptr (y)[x];
            
            if (invDepthConstraint (x, y, w0, w1, row_depthinv, &error_depthinv, &n_factor))
            {
              float error_depthinv_unbiased = error_depthinv - (bias_depthinv / sigma_depthinv);
              weight_depthinv = computeWeightStudent(&error_depthinv_unbiased, nu_depthinv)*__int2float_rn(1 - (weighting == PHOT_ONLY));
            }
         
            if (intensityConstraint (x, y, w0, row_int, &error_int))
            {   
              float error_int_unbiased = error_int - (bias_int / sigma_int);
              weight_int = computeWeightStudent(&error_int_unbiased, nu_int)*__int2float_rn(1 - (weighting == GEOM_ONLY)); 
            }
             
            if (weighting == MIN_WEIGHT)
            {
              weight_int = min(weight_depthinv, weight_int);
            } 
            
            int shift = 0;
            for (int i = 0; i < B_SIZE; ++i)        //rows
            {
              #pragma unroll
              for (int j = i; j < B_SIZE; ++j)          // cols + b
              {   
                local_sys[shift++] += weight_int*(row_int[i] * row_int[j]) + n_factor*weight_depthinv*(row_depthinv[i] * row_depthinv[j]);
              }
              
              local_sys[shift++] += weight_int*(row_int[i] *  (error_int)) + n_factor*weight_depthinv*(row_depthinv[i] * (error_depthinv));
            }
           
          }
        }
        
        __shared__ float smem[CTA_SIZE];       
        unsigned int tid_block = Block::flattenedThreadId ();
        
        for (int i = 0; i < TOTAL_SIZE; ++i)        //rows
        {              
          __syncthreads ();        
          smem[tid_block] = local_sys[i];
          __syncthreads ();

          reduce<CTA_SIZE>(smem);
          
          __syncthreads ();

          if (tid_block == 0)
            gbuf.ptr (i)[blockIdx.x + gridDim.x * blockIdx.y] = smem[0];
        }        
      } 
      
    };
    
    
    
    static __global__ void
    __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
    partialSumKernelGridStride (const constraintsHandler ch) 
    {
      ch.computeSystemGridStride ();
    }  
    
    static __global__ void
    __launch_bounds__(MAX_THREADS_PER_BLOCK, 2)
    partialSumStudentNuKernelGridStride (const constraintsHandler ch) 
    {
      ch.computeStudentNuSystemGridStride ();
    } 
    
    struct FinalReductionHandlerVO
    {
      enum
      {
        CTA_SIZE = MIN_OPTIMAL_THREADS_PER_BLOCK,
        STRIDE = CTA_SIZE,
      };

      PtrStep<float_type> gbuf;
      int length;
      mutable float_type* output;

      __device__ __forceinline__ void
      operator () () const
      {
        const float_type *beg = gbuf.ptr (blockIdx.x);  //1 block per element in A and b
        const float_type *end = beg + length;   //length = num_constraints

        int tid = threadIdx.x;
        
        float_type sum = 0.f;
        for (const float_type *t = beg + tid; t < end; t += STRIDE)  //Each thread sums #(num_contraints/CTA_SIZE) elements
          sum += *t;

        __shared__ float smem[CTA_SIZE];

        smem[tid] = sum;
        __syncthreads ();

        reduce<CTA_SIZE>(smem);
        __syncthreads ();

        if (tid == 0)
          output[blockIdx.x] = smem[0];
      }
    };

    static __global__ void
    FinalReductionKernel (const FinalReductionHandlerVO frh) 
    {
      frh ();
    }

    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float 
    buildSystemGridStride (const float3 delta_trans, const float3 delta_rot,
                 const DepthMapf& W0, const IntensityMapf& I0,
                 const GradientMap& gradW0_x, const GradientMap& gradW0_y, 
                 const GradientMap& gradI0_x, const GradientMap& gradI0_y,		
                 const DepthMapf& W1, const IntensityMapf& I1,	
                 int Mestimator, int weighting,
                 float sigma_depthinv, float sigma_int,
                 float bias_depthinv, float bias_int,
                 const Intr& intr, const int size_A,
                 DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, float_type* matrixA_host, float_type* vectorB_host,
                 int numSMs)
    {
      cudaTimer timer;      

      int cols = W0.cols();
      int rows = W0.rows();

      constraintsHandler ch;

      ch.delta_trans = delta_trans;
      ch.delta_rot = delta_rot;

      ch.W0 = W0;
      ch.gradW0_x = gradW0_x;
      ch.gradW0_y = gradW0_y;

      ch.I0 = I0;
      ch.gradI0_x = gradI0_x;
      ch.gradI0_y = gradI0_y;

      ch.W1 = W1;	
      ch.I1 = I1;

      ch.intr.fx = intr.fx; 
      ch.intr.fy = intr.fy; 
      ch.intr.cx = intr.cx; 
      ch.intr.cy = intr.cy; 

      ch.cols = cols;
      ch.rows = rows;

      ch.Mestimator = Mestimator;
      ch.weighting = weighting;
      ch.sigma_depthinv = sigma_depthinv; //sigma_{u/fb} in [m^-1] ->[m^-1]
      ch.sigma_int = sigma_int;
      ch.bias_depthinv = bias_depthinv;
      ch.bias_int = bias_int;
      
      
      dim3 block (constraintsHandler::CTA_SIZE_X, constraintsHandler::CTA_SIZE_Y);
      
      int gridX = divUp (ch.cols, block.x);
      int gridY = divUp (ch.rows, block.y);
      
      /////////////////////////////////////
      int nThreadsPerBlock = constraintsHandler::CTA_SIZE;
      int numMaxThreadsPerSM, numMaxBlocks, numMaxBlocksPerSM;
      
      //Get device props (5->num SM of gtx850M, 2048->max num of th per MP in Maxwell arch)
      //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
      //cudaDeviceGetAttribute(&numMaxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
      
      if ((numSMs < 1) || (numSMs > dev_prop.multiProcessorCount))
      {
        numSMs = dev_prop.multiProcessorCount;
      }
      //numSMs = dev_prop.multiProcessorCount;
      numMaxThreadsPerSM = dev_prop.maxThreadsPerMultiProcessor;
      
      numMaxBlocksPerSM = std::min(MAX_BLOCKS_PER_SM, divUp(numMaxThreadsPerSM + 1, nThreadsPerBlock) - 1); //Should be always equal, but jut in case
      numMaxBlocks = numSMs*numMaxBlocksPerSM;
      //Above computation with gtx850 and 64 threads/block is 160 (5*2048/64), but 120 performs better (I guess because of massive register/local mem usage??)
      numMaxBlocks = std::min(120, numMaxBlocks); 
      //std::cout << numSMs << std::endl;
       
      //int nPixels = cols*rows;
      
           
      int nBlocksInit = gridX*gridY;
      
      if (nBlocksInit > numMaxBlocks)
      {          
        findBest2DBlockGridCombination(numMaxBlocks, constraintsHandler::CTA_SIZE_X, constraintsHandler::CTA_SIZE_Y, 
                                       ch.cols, ch.rows, gridX, gridY);                                                   
      }
      ///////////////////////////////////
      
      dim3 grid(gridX, gridY);
      
      int nBlocks = gridX*gridY;

      mbuf.create (TOTAL_SIZE);

      if (gbuf.rows () != TOTAL_SIZE || gbuf.cols () != (int)(nBlocks))
        gbuf.create (TOTAL_SIZE, nBlocks);

      ch.gbuf = gbuf;   
      
      cudaFuncSetCacheConfig (partialSumKernelGridStride, cudaFuncCachePreferL1);
      partialSumKernelGridStride<<<grid, block>>>(ch); 
      
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall(cudaStreamSynchronize(0));

      //printFuncAttrib(partialSumKernel_depth);

      FinalReductionHandlerVO frh;
      frh.gbuf = gbuf;
      frh.length = nBlocks;
      frh.output = mbuf;
      
      int blockSizeFinalReduction;
      
      blockSizeFinalReduction = FinalReductionHandlerVO::CTA_SIZE;

      FinalReductionKernel<<<TOTAL_SIZE, blockSizeFinalReduction>>>(frh);
      cudaSafeCall (cudaGetLastError ());
      
      //printFuncAttrib(FinalReductionKernel);
      cudaSafeCall (cudaStreamSynchronize(0));

      float_type host_data[TOTAL_SIZE];
      mbuf.download (host_data);

      int shift = 0;
      for (int i = 0; i < B_SIZE; ++i)  //rows
      {
        for (int j = i; j < B_SIZE + 1; ++j)    // cols + b
        {
          float_type value = host_data[shift++];
          
          if (j == B_SIZE)       // vector b
            vectorB_host[i] = value;
          else
            matrixA_host[j * B_SIZE + i] = matrixA_host[i * B_SIZE + j] = value;
        }
      }
      
      return timer.getTime();
    }; 
    
    
    
    float 
    buildSystemStudentNuGridStride (const float3 delta_trans, const float3 delta_rot,
                 const DepthMapf& W0, const IntensityMapf& I0,
                 const GradientMap& gradW0_x, const GradientMap& gradW0_y, 
                 const GradientMap& gradI0_x, const GradientMap& gradI0_y,		
                 const DepthMapf& W1, const IntensityMapf& I1,	
                 int Mestimator, int weighting,
                 float sigma_depthinv, float sigma_int, 
                 float bias_depthinv, float bias_int, 
                 float nu_depthinv, float nu_int,
                 const Intr& intr, const int size_A,
                 DeviceArray2D<float_type>& gbuf, DeviceArray<float_type>& mbuf, float_type* matrixA_host, float_type* vectorB_host,
                 int numSMs)
    {
      cudaTimer timer;      

      int cols = W0.cols();
      int rows = W0.rows();

      constraintsHandler ch;

      ch.delta_trans = delta_trans;
      ch.delta_rot = delta_rot;

      ch.W0 = W0;
      ch.gradW0_x = gradW0_x;
      ch.gradW0_y = gradW0_y;

      ch.I0 = I0;
      ch.gradI0_x = gradI0_x;
      ch.gradI0_y = gradI0_y;

      ch.W1 = W1;	
      ch.I1 = I1;

      ch.intr.fx = intr.fx; 
      ch.intr.fy = intr.fy; 
      ch.intr.cx = intr.cx; 
      ch.intr.cy = intr.cy; 

      ch.cols = cols;
      ch.rows = rows;
      

      ch.Mestimator = Mestimator;
      ch.weighting = weighting;
      ch.sigma_depthinv = sigma_depthinv; //sigma_{u/fb} in [m^-1] ->[m^-1]
      ch.sigma_int = sigma_int;
      ch.bias_depthinv = bias_depthinv;
      ch.bias_int = bias_int;
      ch.nu_int = nu_int;
      ch.nu_depthinv = nu_depthinv;
      
      /////////////////////////////////////
      int nThreadsPerBlock = constraintsHandler::CTA_SIZE;
      int numMaxThreadsPerSM, numMaxBlocks, numMaxBlocksPerSM;
      
      //Get device props (5->num SM of gtx850M, 2048->max num of th per MP in Maxwell arch)
      //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
      //cudaDeviceGetAttribute(&numMaxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
      if ((numSMs < 1) || (numSMs > dev_prop.multiProcessorCount))
      {
        numSMs = dev_prop.multiProcessorCount;
      }
      //numSMs = dev_prop.multiProcessorCount;
      numMaxThreadsPerSM = dev_prop.maxThreadsPerMultiProcessor;
      
      numMaxBlocksPerSM = std::min(MAX_BLOCKS_PER_SM, divUp(numMaxThreadsPerSM + 1, nThreadsPerBlock) - 1); //Should be always equal, but jut in case
      numMaxBlocks = numSMs*numMaxBlocksPerSM;
      
      
      
      numMaxBlocks = std::min(120, numMaxBlocks); //Above computation with gtx850 and 64 threads/block is 160 (5*2048/64), but 120 performs better (I guess because massive register/local mem usage??)
      //std::cout << numSMs << std::endl;
       
      //int nPixels = cols*rows;
      dim3 block (constraintsHandler::CTA_SIZE_X, constraintsHandler::CTA_SIZE_Y);
      
      int gridX = divUp (ch.cols, block.x);
      int gridY = divUp (ch.rows, block.y);
           
      int nBlocksInit = gridX*gridY;
      
      if (nBlocksInit > numMaxBlocks)
      {          
        findBest2DBlockGridCombination(numMaxBlocks, constraintsHandler::CTA_SIZE_X, constraintsHandler::CTA_SIZE_Y, 
                                       ch.cols, ch.rows, gridX, gridY);                                                   
      }
      //////////////////////////////////////////////
      
      
      dim3 grid(gridX, gridY);
      
      int nBlocks = gridX*gridY;

      mbuf.create (TOTAL_SIZE);

      if (gbuf.rows () != TOTAL_SIZE || gbuf.cols () != (int)(nBlocks))
        gbuf.create (TOTAL_SIZE, nBlocks);

      ch.gbuf = gbuf;   
      
      cudaFuncSetCacheConfig (partialSumStudentNuKernelGridStride, cudaFuncCachePreferL1);
      partialSumStudentNuKernelGridStride<<<grid, block>>>(ch); 
      
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall(cudaStreamSynchronize(0));

      FinalReductionHandlerVO frh;
      frh.gbuf = gbuf;
      frh.length = nBlocks;
      frh.output = mbuf;
      
      int blockSizeFinalReduction;
      
      blockSizeFinalReduction = FinalReductionHandlerVO::CTA_SIZE;

      FinalReductionKernel<<<TOTAL_SIZE, blockSizeFinalReduction>>>(frh);
      cudaSafeCall (cudaGetLastError ());
      
      cudaSafeCall (cudaStreamSynchronize(0));

      float_type host_data[TOTAL_SIZE];
      mbuf.download (host_data);

      int shift = 0;
      for (int i = 0; i < B_SIZE; ++i)  //rows
      {
        for (int j = i; j < B_SIZE + 1; ++j)    // cols + b
        {
          float_type value = host_data[shift++];
          
          if (j == B_SIZE)       // vector b
            vectorB_host[i] = value;
          else
            matrixA_host[j * B_SIZE + i] = matrixA_host[i * B_SIZE + j] = value;
        }
      }
      
      return timer.getTime();
    }; 
    

  }
}

