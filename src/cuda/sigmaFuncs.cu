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
#include <cmath>

namespace RGBID_SLAM
{
  namespace device
  {
    __constant__ float THRESHOLD_HUBER_DEV = 1.345;
    __constant__ float THRESHOLD_TUKEY_DEV = 4.685;
    __constant__ float STUDENT_DOF_DEV = 5.f;

    template<int CTA_SIZE_, typename T>
    static __device__ __forceinline__ void reduce(volatile T* buffer)
    {
      int tid = threadIdx.x;
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
    
    struct errorHandler
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
      
      PtrStep<float> im0;
      PtrStep<float> im1;
      PtrStep<float> N0;

      int cols;
      int rows;	
      
      Intr intr;
      
      int stride_ds;
      
      mutable float* error;
      
      __device__ __forceinline__ void
      computeErrorGridStride () const
      {   
        unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y;  
        unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
        
        for (unsigned int x = tid_x;   x < cols; x += blockDim.x * gridDim.x)    
         {  
          for (unsigned int y = tid_y;   y < rows; y += blockDim.y * gridDim.y)    
          {  
            //Threads in dst
            float value = im1.ptr (stride_ds*y)[stride_ds*x] - im0.ptr (stride_ds*y)[stride_ds*x];
            int error_idx = y*cols + x;
            error[error_idx] = value;	
          }
        }  
        
        return;					    
      }      		
    };
    
    static __global__ void
    normalizeAndAppendErrorsKernel(float* error_int, float* error_depth, float* error_normalised, float sigma_int, float sigma_depth, int size)
    {
      unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      
      for (unsigned int x = tid_x;   x < size; x += blockDim.x * gridDim.x)    
      {  
        error_normalised[x] = error_int[x]/sigma_int;
        error_normalised[x+size] = error_depth[x]/sigma_depth;
      }
      
      return;
      
    }
  	
      struct sigmaHandler
      {
        enum
        {
          CTA_SIZE = MIN_OPTIMAL_THREADS_PER_BLOCK,
        };

        float* error;

        int nel;   
             
        int Mestimator;
        float sigma;
        float bias;
        float nu;
        mutable float* moments_dev;
        mutable float* func_weights_nu_dev;
        
        mutable float* global_weighted_sq_res;
        mutable float* global_weighted_res;
        mutable float* global_weights;
        mutable float* global_nel;
        
        mutable float* global_ln_weights;
        
        int aux_size;
      
        __device__ __forceinline__ void
        partialBiasAndSigma() const
        {
          const float *beg = &error[0];  
          const float *end = beg + nel;  

          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          
          __shared__ float smem_weighted_sq_res[CTA_SIZE];
          __shared__ float smem_weighted_res[CTA_SIZE];
          __shared__ float smem_weights[CTA_SIZE];
          __shared__ float smem_nel[CTA_SIZE];
          __syncthreads ();
                    
          float sum_weighted_sq_res = 0.f;
          float sum_weighted_res = 0.f;
          float sum_weights = 0.f;
          float sum_nel = 0.f;

          for (const float *t = beg + tid; t < end; t += blockDim.x*gridDim.x) 
          { 
            float is_valid = 0.f;
            float weighted_sq_res = 0.f; 
            float weighted_res = 0.f; 
            float weight = 0.f; 
            float error_raw = *t;

            if ( !(isinf(error_raw)) && !(isnan(error_raw)))
            {		          
              weight = 1.f;
              is_valid = 1.f;
              
              float error_unbias = error_raw - bias;
              float error_normalised = error_unbias/sigma;

              if ( (Mestimator == HUBER) && (fabs(error_normalised) > THRESHOLD_HUBER_DEV))
              {
                weight = (THRESHOLD_HUBER_DEV / fabs(error_normalised));
              }
              else if (Mestimator == TUKEY)
              {
                if (fabs(error_normalised) < THRESHOLD_TUKEY_DEV)
                {
                  float aux1 = ( error_normalised / THRESHOLD_TUKEY_DEV ) * ( error_normalised / THRESHOLD_TUKEY_DEV );
                  weight = (1.f - aux1) * (1.f - aux1);		
                }
                else
                {
                weight = 0.f;	

                //not what theory says, but empirically the performance is better. 
                /*Maybe it is related with the fact that if one allows Tukey estimator to be defined in (-infty, infty) 
                pdf's normalizing const = infty so that integral(pdf) = 1, and loglikelihood = infty for every sigma. 
                Hence the need of cutting tails  of Tukey's associated "pdf" to have a real pdf*/
                is_valid = 0.f; 

                }
              }
              else if (Mestimator == STUDENT)
              {
                weight = (STUDENT_DOF_DEV + 1.f) / (STUDENT_DOF_DEV + error_normalised*error_normalised);             
              }

              weighted_res = error_raw*weight;
              weighted_sq_res = weighted_res*error_raw;
            }

            sum_weighted_sq_res += weighted_sq_res;
            sum_weighted_res += weighted_res;
            sum_weights += weight;
            sum_nel += is_valid; 
          }

          //__syncthreads ();
          smem_weighted_sq_res[threadIdx.x] = sum_weighted_sq_res;
          smem_weighted_res[threadIdx.x] = sum_weighted_res;
          smem_weights[threadIdx.x] = sum_weights;
          smem_nel[threadIdx.x] = sum_nel;
          
          __syncthreads ();

          reduce<CTA_SIZE>(smem_weighted_sq_res);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_weighted_res);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_weights);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_nel);
          __syncthreads (); //I think no need to sync since warp of tid = 0 is the last in finishing the reduction         
          
          if (threadIdx.x == 0)
          {
            global_weighted_sq_res[blockIdx.x] = smem_weighted_sq_res[0];
            global_weighted_res[blockIdx.x] = smem_weighted_res[0];
            global_weights[blockIdx.x] = smem_weights[0];
            global_nel[blockIdx.x] = smem_nel[0];
          }
				        
		      return;
        }  
        
        
        __device__ __forceinline__ void
        partialBiasAndSigmaStudent() const
        {
          const float *beg = &error[0];  
          const float *end = beg + nel;  

          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          
          __shared__ float smem_weighted_sq_res[CTA_SIZE];
          __shared__ float smem_weighted_res[CTA_SIZE];
          __shared__ float smem_weights[CTA_SIZE];
          __shared__ float smem_nel[CTA_SIZE];
          __syncthreads ();
                    
          float sum_weighted_sq_res = 0.f;
          float sum_weighted_res = 0.f;
          float sum_weights = 0.f;
          float sum_nel = 0.f;

          for (const float *t = beg + tid; t < end; t += blockDim.x*gridDim.x) 
          { 
            float is_valid = 0.f;
            float weighted_sq_res = 0.f; 
            float weighted_res = 0.f; 
            float weight = 0.f; 
            float error_raw = *t;

            if ( !(isinf(error_raw)) && !(isnan(error_raw)))
            {		 
              is_valid = 1.f;
              
              if (Mestimator == LSQ)
              {
                weight = 1.f;
              }
              else
              {
                float error_unbias = error_raw - bias;
                float error_normalised = error_unbias/sigma;

                weight = (nu + 1.f) / (nu + error_normalised*error_normalised); 
              }

              weighted_res = error_raw*weight;
              weighted_sq_res = weighted_res*error_raw;
            }

            sum_weighted_sq_res += weighted_sq_res;
            sum_weighted_res += weighted_res;
            sum_weights += weight;
            sum_nel += is_valid; 
          }

          //__syncthreads ();
          smem_weighted_sq_res[threadIdx.x] = sum_weighted_sq_res;
          smem_weighted_res[threadIdx.x] = sum_weighted_res;
          smem_weights[threadIdx.x] = sum_weights;
          smem_nel[threadIdx.x] = sum_nel;
          
          __syncthreads ();

          reduce<CTA_SIZE>(smem_weighted_sq_res);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_weighted_res);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_weights);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_nel);
          __syncthreads (); //I think no need to sync since warp of tid = 0 is the last in finishing the reduction         
          
          if (threadIdx.x == 0)
          {
            global_weighted_sq_res[blockIdx.x] = smem_weighted_sq_res[0];
            global_weighted_res[blockIdx.x] = smem_weighted_res[0];
            global_weights[blockIdx.x] = smem_weights[0];
            global_nel[blockIdx.x] = smem_nel[0];
          }
				        
		      return;
        }  
        
        __device__ __forceinline__ void
        finalReductionBiasAndSigma() const
        {
          int tid = threadIdx.x;
          
          __shared__ float smem_sum_global_weighted_sq_res[CTA_SIZE];
          __shared__ float smem_sum_global_weighted_res[CTA_SIZE];
          __shared__ float smem_sum_global_weights[CTA_SIZE];
          __shared__ float smem_sum_global_nel[CTA_SIZE];

          float sum_global_weighted_sq_res = 0.f;
          float sum_global_weighted_res = 0.f;
          float sum_global_weights = 0.f;
          float sum_global_nel = 0.f;
          
          for (int i = tid; i < aux_size; i += blockDim.x)  
          {
            sum_global_weighted_sq_res += global_weighted_sq_res[i];
            sum_global_weighted_res += global_weighted_res[i];
            sum_global_weights += global_weights[i];
            sum_global_nel += global_nel[i];
          }
          
          smem_sum_global_weighted_sq_res[tid] = sum_global_weighted_sq_res;
          smem_sum_global_weighted_res[tid] = sum_global_weighted_res;
          smem_sum_global_weights[tid] = sum_global_weights;
          smem_sum_global_nel[tid] = sum_global_nel;
          
          __syncthreads ();

          reduce<CTA_SIZE>(smem_sum_global_weighted_sq_res);   
          __syncthreads ();
          reduce<CTA_SIZE>(smem_sum_global_weighted_res);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_sum_global_weights);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_sum_global_nel);   
          __syncthreads ();    
          
          if (tid == 0)
          {
            moments_dev[0] = smem_sum_global_weighted_res[0] / smem_sum_global_weights[0];
            moments_dev[1] = sqrt((smem_sum_global_weighted_sq_res[0] - 2.f*moments_dev[0]*smem_sum_global_weighted_res[0] + moments_dev[0]*moments_dev[0]*smem_sum_global_weights[0]) / smem_sum_global_nel[0]);
            //moments_dev[1] = sqrt((smem_sum_global_weighted_sq_res[0]) / smem_sum_global_weights[0]);
          }
           
          return;
        }
        
        
        __device__ __forceinline__ void
        partialFuncWeightsNu() const
        {
          const float *beg = &error[0];  
          const float *end = beg + nel;  

          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          
          __shared__ float smem_ln_weights[CTA_SIZE];
          __shared__ float smem_weights[CTA_SIZE];
          __shared__ float smem_nel[CTA_SIZE];
          __syncthreads ();
          
          float sum_ln_weights = 0.f;
          float sum_weights = 0.f;
          float sum_nel = 0.f;

          for (const float *t = beg + tid; t < end; t += blockDim.x*gridDim.x) 
          { 
            float is_valid = 0.f;
            float ln_weight = 0.f;
            float weight = 0.f; 
            float error_raw = *t;

            if ( !(isinf(error_raw)) && !(isnan(error_raw)))
            { 
              is_valid = 1.f;
              
              float error_unbias = error_raw - bias;
              float error_normalised = error_unbias/sigma;

              weight = (nu + 1.f) / (nu + error_normalised*error_normalised); 
              ln_weight = log(weight);
            }

            sum_ln_weights += ln_weight;
            sum_weights += weight;
            sum_nel += is_valid; 
          }

          //__syncthreads ();
          //smem_weighted_sq_res[threadIdx.x] = sum_weighted_sq_res;
          smem_ln_weights[threadIdx.x] = sum_ln_weights;
          smem_weights[threadIdx.x] = sum_weights;
          smem_nel[threadIdx.x] = sum_nel;
          
          __syncthreads ();

          reduce<CTA_SIZE>(smem_ln_weights);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_weights);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_nel);
          __syncthreads (); //I think no need to sync since warp of tid = 0 is the last in finishing the reduction         
          
          if (threadIdx.x == 0)
          {
            global_ln_weights[blockIdx.x] = smem_ln_weights[0];
            global_weights[blockIdx.x] = smem_weights[0];
            global_nel[blockIdx.x] = smem_nel[0];
          }
				        
		      return;
        }  
        
        
        __device__ __forceinline__ void
        finalReductionFuncWeightsNu() const
        {
          int tid = threadIdx.x;          
          
          __shared__ float smem_sum_global_ln_weights[CTA_SIZE];
          __shared__ float smem_sum_global_weights[CTA_SIZE];
          __shared__ float smem_sum_global_nel[CTA_SIZE];

          float sum_global_ln_weights = 0.f;
          float sum_global_weights = 0.f;
          float sum_global_nel = 0.f;
          
          for (int i = tid; i < aux_size; i += blockDim.x)  
          {            
            sum_global_ln_weights += global_ln_weights[i];
            sum_global_weights += global_weights[i];
            sum_global_nel += global_nel[i];
          }
          
          smem_sum_global_ln_weights[tid] = sum_global_ln_weights;
          smem_sum_global_weights[tid] = sum_global_weights;
          smem_sum_global_nel[tid] = sum_global_nel;
          
          __syncthreads ();
          reduce<CTA_SIZE>(smem_sum_global_ln_weights);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_sum_global_weights);
          __syncthreads ();
          reduce<CTA_SIZE>(smem_sum_global_nel);   
          __syncthreads ();    
          
          if (tid == 0)
          {
            func_weights_nu_dev[0] = (smem_sum_global_ln_weights[0] - smem_sum_global_weights[0])/ smem_sum_global_nel[0];
          }
           
          return;
        }
        
      };

		
		  struct chiSquaredHandler
      {
        enum
        {
          CTA_SIZE = 256
        };

        float* error;

        int Mestimator;
        float sigma;
        int nel;
        int aux_size;
        
        mutable float* global_rho;
        mutable float* global_nel;

        mutable float* chi_squared_dev;
        mutable float* Ndof_dev;

        __device__ __forceinline__ void
        partialChiSquared () const
        {
          const float *beg = &error[0];  
          const float *end = beg + nel;  

          int tid = threadIdx.x + blockIdx.x * blockDim.x;
          
          __shared__ float smem_N[CTA_SIZE];
          __shared__ float smem_rho[CTA_SIZE];
          __syncthreads ();
                    
          float sum_N = 0.f;
          float sum_rho = 0.f;

          for (const float *t = beg + tid; t < end; t += blockDim.x*gridDim.x) 
          { 
            float is_valid = 0.f;
            float rho = 0.f;
            float error_normalised = *t;

            if ( !(isinf(error_normalised)) && !(isnan(error_normalised)))
            {	  
              is_valid = 1.f;
              rho = (error_normalised*error_normalised)/2.f;

              if ( (Mestimator == HUBER) && (fabs(error_normalised) > THRESHOLD_HUBER_DEV))
              {                
                rho = THRESHOLD_HUBER_DEV*(fabs(error_normalised) - THRESHOLD_HUBER_DEV/2.f);
              }
              else if (Mestimator == TUKEY)
              {
                if (fabs(error_normalised) < THRESHOLD_TUKEY_DEV)
                {
                  float aux1 = ( error_normalised / THRESHOLD_TUKEY_DEV ) * ( error_normalised / THRESHOLD_TUKEY_DEV );
                  float aux2 = (1.f-aux1)*(1.f-aux1)*(1.f-aux1);
                  rho = ((THRESHOLD_TUKEY_DEV*THRESHOLD_TUKEY_DEV)/6.f)*(1.f-aux2);
                }
                else
                {
                  rho = ((THRESHOLD_TUKEY_DEV*THRESHOLD_TUKEY_DEV)/6.f);
                }
              }
              else if (Mestimator == STUDENT)
              {
                rho = ((STUDENT_DOF_DEV + 1.f) /  2.f)*log(1.f + (error_normalised*error_normalised)/STUDENT_DOF_DEV); 
                //float weight = (STUDENT_DOF_DEV + 1.f) / (STUDENT_DOF_DEV + (error_normalised)*(error_normalised));
                //rho = weight*rho;            
              }  
            }
            
            sum_N += is_valid;
            sum_rho += rho;
          }

          __syncthreads ();
          smem_N[threadIdx.x] = sum_N;
          smem_rho[threadIdx.x] = sum_rho;     
          __syncthreads ();

          reduce<CTA_SIZE>(smem_N);	__syncthreads ();
          reduce<CTA_SIZE>(smem_rho);	__syncthreads ();					

          if (threadIdx.x == 0)
          {
            global_nel[blockIdx.x] = smem_N[0];
            global_rho[blockIdx.x] = smem_rho[0];
          }
          
          return;			
        }
        
         __device__ __forceinline__ void
        finalReductionChiSquared() const
        {
          int tid = threadIdx.x;
          
          __shared__ float smem_sum_global_rho[CTA_SIZE];
          __shared__ float smem_sum_global_nel[CTA_SIZE];

          float sum_global_rho = 0.f;
          float sum_global_nel = 0.f;
          
          for (int i = tid; i < aux_size; i += blockDim.x)  
          {
            sum_global_rho += global_rho[i];
            sum_global_nel += global_nel[i];
          }
          
          smem_sum_global_rho[tid] = sum_global_rho;
          smem_sum_global_nel[tid] = sum_global_nel;
          
          __syncthreads ();

          reduce<CTA_SIZE>(smem_sum_global_rho);   
          __syncthreads ();
          reduce<CTA_SIZE>(smem_sum_global_nel);   
          __syncthreads ();    
          
          if (tid == 0)
          {
            chi_squared_dev[0] = smem_sum_global_rho[0] / smem_sum_global_nel[0];
            Ndof_dev[0] = smem_sum_global_nel[0];
          }
           
          return;
        }
      };
      
      
      static __global__ void
      errorGridStrideKernel (const errorHandler eh) 
      {
        eh.computeErrorGridStride ();
      }

      static __global__ void
      finalReductionBiasAndSigmaKernel (const sigmaHandler sh) 
      {
        sh.finalReductionBiasAndSigma ();
      }

      static __global__ void
      computeBiasAndSigmaKernelPartial (const sigmaHandler sh) 
      {
        sh.partialBiasAndSigma ();
      }
      
      static __global__ void
      computeBiasAndSigmaStudentKernelPartial (const sigmaHandler sh) 
      {
        sh.partialBiasAndSigmaStudent ();
      }
      
      static __global__ void
      finalReductionFuncWeightsNuKernel (const sigmaHandler sh) 
      {
        sh.finalReductionFuncWeightsNu ();
      }
      
      static __global__ void
      computeFuncWeightsNuKernelPartial (const sigmaHandler sh) 
      {
        sh.partialFuncWeightsNu ();
      }
      
      static __global__ void
      finalReductionChiSquaredKernel (const chiSquaredHandler csh) 
      {
        csh.finalReductionChiSquared ();
      }

      static __global__ void
      computeChiSquaredKernelPartial (const chiSquaredHandler csh) 
      {
        csh.partialChiSquared ();
      }
      
      
      //////////////////////////////////////////////////////////////
      float 
      computeErrorGridStride (const DeviceArray2D<float>& im1,  const DeviceArray2D<float>& im0,  DeviceArray<float>& error, int min_Nsamples, int numSMs)
      {
        cudaTimer timer;      
        
        errorHandler eh;

        eh.im0 = im0;
        eh.im1 = im1;

        int error_size = im0.cols()*im0.rows();
        int cols_prev = im0.cols();
        int rows_prev = im0.rows();
          
        //std::cout << "Start compute error: " <<  std::endl;
        
        if (min_Nsamples < error_size)
        {
          int cols_curr;
          int rows_curr;
          
          while(1)
          {
            cols_curr = cols_prev/2;
            rows_curr = rows_prev/2;
            
            if ( ((2*cols_curr-cols_prev) != 0) || ((2*rows_curr-rows_prev) != 0) || (min_Nsamples > cols_curr*rows_curr) )
            {
              error_size = cols_prev*rows_prev;
              break;
            }
            
            cols_prev = cols_curr;
            rows_prev = rows_curr;
          }
        }
        
        error.create(error_size);        
        
        int stride =  sqrt((im0.rows()*im0.cols()) / error_size);
        assert(stride*stride == ((im0.rows()*im0.cols()) / error_size));
        
        eh.cols = cols_prev;
        eh.rows = rows_prev;

        eh.error = error;
        eh.stride_ds = stride;
        
        dim3 block (errorHandler::CTA_SIZE_X, errorHandler::CTA_SIZE_Y);
        
        int gridX = divUp (eh.cols, block.x);
        int gridY = divUp (eh.rows, block.y);
        
        computeBlockDim2D(gridX, gridY, block.x, block.y, eh.cols, eh.rows, numSMs);
        
        dim3 grid(gridX, gridY);

        cudaFuncSetCacheConfig (errorGridStrideKernel, cudaFuncCachePreferL1);

        errorGridStrideKernel<<<grid, block>>>(eh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaStreamSynchronize(0));

        return timer.getTime();
      };
      
      ////////////////////////////////////
      
      

      
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
      float  
      computeSigmaPdf (DeviceArray<float>& error, float &bias, float &sigma, int Mestimator, int numSMs)
      {        
        cudaTimer timer;     

        sigmaHandler sh;
        
        sh.error = error;
        sh.nel = error.size();	
        
        int grid = divUp(sh.nel, sigmaHandler::CTA_SIZE); 
        
        computeBlockDim1D(grid, sigmaHandler::CTA_SIZE, numSMs);
        
        DeviceArray<float> moments_dev;
        moments_dev.create(2);  
        sh.moments_dev = moments_dev;   
        
        DeviceArray<float> global_weighted_sq_res;
        global_weighted_sq_res.create(grid);  
        sh.global_weighted_sq_res = global_weighted_sq_res;  
        
        DeviceArray<float> global_weighted_res;
        global_weighted_res.create(grid);  
        sh.global_weighted_res = global_weighted_res;
        
        DeviceArray<float> global_weights;
        global_weights.create(grid);  
        sh.global_weights = global_weights;
        
        DeviceArray<float> global_nel;
        global_nel.create(grid);  
        sh.global_nel = global_nel;
        
        sh.aux_size = grid;
        
        //////////////////////////////
        //setting init bias and sigma (conv. guaranteed for high init guess??)
        sh.sigma = sigma;
        sh.bias = bias;
        sh.Mestimator = LSQ; 
        
        int max_iters = 10;
        float rel_tol = 0.1;
        
        for (int i=0; i < max_iters; i++)
        {
          float moments_host[2];
          cudaFuncSetCacheConfig (computeBiasAndSigmaKernelPartial, cudaFuncCachePreferShared);      
          computeBiasAndSigmaKernelPartial<<<grid,sigmaHandler::CTA_SIZE>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaStreamSynchronize(0));
          
          cudaFuncSetCacheConfig (finalReductionBiasAndSigmaKernel, cudaFuncCachePreferShared);
          finalReductionBiasAndSigmaKernel<<<1,sigmaHandler::CTA_SIZE>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaStreamSynchronize(0));
          
          moments_dev.download(moments_host);
        
          bias = moments_host[0];
          sigma = moments_host[1];
          
         if ((i > 0) && ((abs(sigma - sh.sigma)/sh.sigma) < rel_tol))
              break;
              
          sh.bias = bias;    
          sh.sigma = sigma;
          sh.Mestimator = Mestimator;
        }
        
        moments_dev.release();
        global_weighted_sq_res.release();
        global_weighted_res.release(); 
        global_weights.release();
        global_nel.release();
        
        //error_sampled.release();
        
        return timer.getTime();

      };
      
      
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
      float  
      computeSigmaAndNuStudent (DeviceArray<float>& error, float &bias, float &sigma, float &nu, int Mestimator, int numSMs)
      {        
        cudaTimer timer;     

        sigmaHandler sh;
        
        sh.error = error;
        sh.nel = error.size();	
        
        int grid = divUp(sh.nel, sigmaHandler::CTA_SIZE); 
        
        computeBlockDim1D(grid, sigmaHandler::CTA_SIZE, numSMs);
        
        DeviceArray<float> moments_dev;
        moments_dev.create(2);  
        sh.moments_dev = moments_dev;   
        
        DeviceArray<float> func_weights_nu_dev;
        func_weights_nu_dev.create(1);  
        sh.func_weights_nu_dev = func_weights_nu_dev; 
        
        DeviceArray<float> global_weighted_sq_res;
        global_weighted_sq_res.create(grid);  
        sh.global_weighted_sq_res = global_weighted_sq_res;  
        
        DeviceArray<float> global_weighted_res;
        global_weighted_res.create(grid);  
        sh.global_weighted_res = global_weighted_res;
        
        DeviceArray<float> global_weights;
        global_weights.create(grid);  
        sh.global_weights = global_weights;
        
        DeviceArray<float> global_ln_weights;
        global_ln_weights.create(grid);  
        sh.global_ln_weights = global_ln_weights;
        
        DeviceArray<float> global_nel;
        global_nel.create(grid);  
        sh.global_nel = global_nel;
        
        sh.aux_size = grid;
        
        //////////////////////////////
        //setting init bias and sigma (conv. guaranteed for high init guess??)
        sh.sigma = sigma;
        sh.bias = bias;
        sh.Mestimator = LSQ; 
        sh.nu = 5.f;
        float sigma_prev = sh.sigma;
        
        int max_iters = 10;
        float rel_tol = 0.1;        
        
        for (int i=0; i < max_iters; i++)
        {          
          float moments_host[2];
          
          cudaFuncSetCacheConfig (computeBiasAndSigmaStudentKernelPartial, cudaFuncCachePreferShared);      
          computeBiasAndSigmaStudentKernelPartial<<<grid,sigmaHandler::CTA_SIZE>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaStreamSynchronize(0));
          
          cudaFuncSetCacheConfig (finalReductionBiasAndSigmaKernel, cudaFuncCachePreferShared);
          finalReductionBiasAndSigmaKernel<<<1,sigmaHandler::CTA_SIZE>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaStreamSynchronize(0));
          
          moments_dev.download(moments_host);
        
          bias = moments_host[0];
          sigma = moments_host[1];
          
          //std::cout << sigma << std::endl,
          sigma_prev = sh.sigma;
             
          sh.bias = bias;    
          sh.sigma = sigma;
          sh.Mestimator = Mestimator;
          
          if ((i > 0) && ((abs(sigma - sigma_prev)/sigma_prev) < rel_tol))
              break;
        }
        
        
        //////Estimate  nu
            
          float func_weights_nu_host[1];
          float nu_up = 10.f;
          float nu_down = 2.f;
          float nu_new; 
          
          sh.nu = nu_down;
          
          cudaFuncSetCacheConfig (computeFuncWeightsNuKernelPartial, cudaFuncCachePreferShared);      
          computeFuncWeightsNuKernelPartial<<<grid,sigmaHandler::CTA_SIZE>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaStreamSynchronize(0));
          
          
          cudaFuncSetCacheConfig (finalReductionFuncWeightsNuKernel, cudaFuncCachePreferShared);
          finalReductionFuncWeightsNuKernel<<<1,sigmaHandler::CTA_SIZE>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaStreamSynchronize(0));
          
          func_weights_nu_dev.download(func_weights_nu_host);
          
          float C_nu_down = -digamma_func(nu_down/2.f) + std::log(nu_down/2.f) + func_weights_nu_host[0] + 1.f + digamma_func((nu_down+1.f)/2.f) - std::log((nu_down+1.f)/2.f);
          
          sh.nu = nu_up;
          
          cudaFuncSetCacheConfig (computeFuncWeightsNuKernelPartial, cudaFuncCachePreferShared);      
          computeFuncWeightsNuKernelPartial<<<grid,sigmaHandler::CTA_SIZE>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaStreamSynchronize(0));
          
          
          cudaFuncSetCacheConfig (finalReductionFuncWeightsNuKernel, cudaFuncCachePreferShared);
          finalReductionFuncWeightsNuKernel<<<1,sigmaHandler::CTA_SIZE>>>(sh);
          cudaSafeCall ( cudaGetLastError () );
          cudaSafeCall(cudaStreamSynchronize(0));
          
          func_weights_nu_dev.download(func_weights_nu_host);
          
          float C_nu_up = -digamma_func(nu_up/2.f) + std::log(nu_up/2.f) + func_weights_nu_host[0] + 1.f + digamma_func((nu_up+1.f)/2.f) - std::log((nu_up+1.f)/2.f);
          
          //C_nu looks like this
          //|*
          //|*
          //|*
          //|*
          //| *
          //| *
          //| *
          //|  *
          //|  *
          //|  *
          //|   *
          //|   *
          //|   *
          //--------------------------------------------------------
          //|    *          *******************************************
          //|     *    ****
          //|       ***
          
          if (C_nu_up*C_nu_down > 0)
          {
            if (C_nu_down <= 0.f)
              sh.nu = nu_down;
            else
              sh.nu = nu_up;
          }     
          else
          {
            for (int j=0; j<5; j++)
            {        
              nu_new = (nu_up+nu_down) / 2; 
              if ((nu_up-nu_down) < 1.f)
                break;   
                     
              sh.nu = nu_new;
              cudaFuncSetCacheConfig (computeFuncWeightsNuKernelPartial, cudaFuncCachePreferShared);      
              computeFuncWeightsNuKernelPartial<<<grid,sigmaHandler::CTA_SIZE>>>(sh);
              cudaSafeCall ( cudaGetLastError () );
              cudaSafeCall(cudaStreamSynchronize(0));
              
              
              cudaFuncSetCacheConfig (finalReductionFuncWeightsNuKernel, cudaFuncCachePreferShared);
              finalReductionFuncWeightsNuKernel<<<1,sigmaHandler::CTA_SIZE>>>(sh);
              cudaSafeCall ( cudaGetLastError () );
              cudaSafeCall(cudaStreamSynchronize(0));
              
              func_weights_nu_dev.download(func_weights_nu_host);
              
              float C_nu_new = -digamma_func(nu_new/2.f) + std::log(nu_new/2.f) + func_weights_nu_host[0] + 1.f + digamma_func((nu_new+1.f)/2.f) - std::log((nu_new+1.f)/2.f);
          
              if (C_nu_new * C_nu_up > 0)
              {
                C_nu_up = C_nu_new;
                nu_up = nu_new;
              }
              else
              {
                C_nu_down = C_nu_new;
                nu_down = nu_new;                
              }
            }
            sh.nu = nu_new;
          }
          /////////////////////////////////
          
        nu = sh.nu;
       //nu = 8.f;
        
        //std::cout << "Nu is              : " << sh.nu << std::endl;
        moments_dev.release();
        func_weights_nu_dev.release();
        global_weighted_sq_res.release();
        global_weighted_res.release(); 
        global_weights.release();
        global_ln_weights.release();
        global_nel.release();
        
        //error_sampled.release();
        
        return timer.getTime();

      };
      
      float  
      computeNuStudent (DeviceArray<float>& error, float &bias, float &sigma, float &nu, int numSMs)
      {
        cudaTimer timer;     

        sigmaHandler sh;
        
        sh.error = error;
        sh.nel = error.size();	
        
        int grid = divUp(sh.nel, sigmaHandler::CTA_SIZE); 
        
        computeBlockDim1D(grid, sigmaHandler::CTA_SIZE, numSMs);
        
        
        DeviceArray<float> func_weights_nu_dev;
        func_weights_nu_dev.create(1);  
        sh.func_weights_nu_dev = func_weights_nu_dev; 
        
        DeviceArray<float> global_weights;
        global_weights.create(grid);  
        sh.global_weights = global_weights;
        
        DeviceArray<float> global_ln_weights;
        global_ln_weights.create(grid);  
        sh.global_ln_weights = global_ln_weights;
        
        DeviceArray<float> global_nel;
        global_nel.create(grid);  
        sh.global_nel = global_nel;
        
        sh.aux_size = grid;
        
        sh.sigma = sigma;
        sh.bias = bias;
                
        //////Estimate  nu
            
        float func_weights_nu_host[1];
        float nu_up = 10.f;
        float nu_down = 2.f;
        float nu_new; 
        
        sh.nu = nu_down;
        
        cudaFuncSetCacheConfig (computeFuncWeightsNuKernelPartial, cudaFuncCachePreferShared);      
        computeFuncWeightsNuKernelPartial<<<grid,sigmaHandler::CTA_SIZE>>>(sh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaStreamSynchronize(0));
        
        
        cudaFuncSetCacheConfig (finalReductionFuncWeightsNuKernel, cudaFuncCachePreferShared);
        finalReductionFuncWeightsNuKernel<<<1,sigmaHandler::CTA_SIZE>>>(sh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaStreamSynchronize(0));
        
        func_weights_nu_dev.download(func_weights_nu_host);
        
        float C_nu_down = -digamma_func(nu_down/2.f) + std::log(nu_down/2.f) + func_weights_nu_host[0] + 1.f + digamma_func((nu_down+1.f)/2.f) - std::log((nu_down+1.f)/2.f);
        
        sh.nu = nu_up;
        
        cudaFuncSetCacheConfig (computeFuncWeightsNuKernelPartial, cudaFuncCachePreferShared);      
        computeFuncWeightsNuKernelPartial<<<grid,sigmaHandler::CTA_SIZE>>>(sh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaStreamSynchronize(0));
        
        
        cudaFuncSetCacheConfig (finalReductionFuncWeightsNuKernel, cudaFuncCachePreferShared);
        finalReductionFuncWeightsNuKernel<<<1,sigmaHandler::CTA_SIZE>>>(sh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaStreamSynchronize(0));
        
        func_weights_nu_dev.download(func_weights_nu_host);
        
        float C_nu_up = -digamma_func(nu_up/2.f) + std::log(nu_up/2.f) + func_weights_nu_host[0] + 1.f + digamma_func((nu_up+1.f)/2.f) - std::log((nu_up+1.f)/2.f);
        
        //C_nu looks like this
        //|*
        //|*
        //|*
        //|*
        //| *
        //| *
        //| *
        //|  *
        //|  *
        //|  *
        //|   *
        //|   *
        //|   *
        //--------------------------------------------------------
        //|    *          *******************************************
        //|     *    ****
        //|       ***
        
        if (C_nu_up*C_nu_down > 0)
        {
          if (C_nu_down <= 0.f)
            sh.nu = nu_down;
          else
            sh.nu = nu_up;
        }     
        else
        {
          for (int j=0; j<5; j++)
          {        
            nu_new = (nu_up+nu_down) / 2; 
            if ((nu_up-nu_down) < 1.f)
              break;   
                   
            sh.nu = nu_new;
            cudaFuncSetCacheConfig (computeFuncWeightsNuKernelPartial, cudaFuncCachePreferShared);      
            computeFuncWeightsNuKernelPartial<<<grid,sigmaHandler::CTA_SIZE>>>(sh);
            cudaSafeCall ( cudaGetLastError () );
            cudaSafeCall(cudaStreamSynchronize(0));
            
            
            cudaFuncSetCacheConfig (finalReductionFuncWeightsNuKernel, cudaFuncCachePreferShared);
            finalReductionFuncWeightsNuKernel<<<1,sigmaHandler::CTA_SIZE>>>(sh);
            cudaSafeCall ( cudaGetLastError () );
            cudaSafeCall(cudaStreamSynchronize(0));
            
            func_weights_nu_dev.download(func_weights_nu_host);
            
            float C_nu_new = -digamma_func(nu_new/2.f) + std::log(nu_new/2.f) + func_weights_nu_host[0] + 1.f + digamma_func((nu_new+1.f)/2.f) - std::log((nu_new+1.f)/2.f);
        
            if (C_nu_new * C_nu_up > 0)
            {
              C_nu_up = C_nu_new;
              nu_up = nu_new;
            }
            else
            {
              C_nu_down = C_nu_new;
              nu_down = nu_new;                
            }
          }
          sh.nu = nu_new;
        }
        /////////////////////////////////
          
        nu = sh.nu;
        
        //std::cout << "Nu is              : " << sh.nu << std::endl;
        func_weights_nu_dev.release();
        global_weights.release();
        global_ln_weights.release();
        global_nel.release();
        
        //error_sampled.release();
        
        return timer.getTime();

      };
      
      
      float 
      computeChiSquare (DeviceArray<float>& error_int,  DeviceArray<float>& error_depth,  float sigma_int, float sigma_depth, int Mestimator, float &chi_squared, float &chi_test, float &Ndof, int numSMs)
      {
        cudaTimer timer;     
      
        DeviceArray<float> error_normalised;
        error_normalised.create(error_depth.size() + error_int.size());  
        
        assert(error_depth.size() == error_int.size());
        
        int grid = divUp(error_depth.size(), chiSquaredHandler::CTA_SIZE);         
        computeBlockDim1D(grid, chiSquaredHandler::CTA_SIZE, numSMs);
        
        cudaFuncSetCacheConfig (normalizeAndAppendErrorsKernel, cudaFuncCachePreferL1);
        normalizeAndAppendErrorsKernel<<<grid, chiSquaredHandler::CTA_SIZE>>>(error_int, error_depth, error_normalised, sigma_int, sigma_depth, error_depth.size());
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaStreamSynchronize(0));
        
        chiSquaredHandler csh;
        
        DeviceArray<float> chi_squared_dev;
        chi_squared_dev.create(1);  
        csh.chi_squared_dev = chi_squared_dev;  
        
        DeviceArray<float> Ndof_dev;
        Ndof_dev.create(1);  
        csh.Ndof_dev = Ndof_dev;  
        
        DeviceArray<float> global_nel;
        global_nel.create(grid);  
        csh.global_nel = global_nel;  
        
        DeviceArray<float> global_rho;
        global_rho.create(grid);  
        csh.global_rho = global_rho;
        
        csh.error = error_normalised;
        csh.Mestimator = Mestimator;
        csh.aux_size = grid;
        csh.nel = error_normalised.size();
        
        cudaFuncSetCacheConfig (computeChiSquaredKernelPartial, cudaFuncCachePreferShared);
        computeChiSquaredKernelPartial<<<grid, chiSquaredHandler::CTA_SIZE>>>(csh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaStreamSynchronize(0));
        
        cudaFuncSetCacheConfig (finalReductionChiSquaredKernel, cudaFuncCachePreferShared);
        finalReductionChiSquaredKernel<<<1,chiSquaredHandler::CTA_SIZE>>>(csh);
        cudaSafeCall ( cudaGetLastError () );
        cudaSafeCall(cudaStreamSynchronize(0));
        
        float host_chi_squared[1];
        
        chi_squared_dev.download(host_chi_squared);
        chi_squared = host_chi_squared[0];
        
        float host_Ndof[1];
        
        Ndof_dev.download(host_Ndof);
        Ndof = host_Ndof[0];
        
        float z_gauss = (chi_squared - Ndof ) / (sqrt(2.f*Ndof));
        chi_test =  0.5f*(1.f + erf(z_gauss / sqrt(2.f)));

        error_normalised.release();
        chi_squared_dev.release();
        Ndof_dev.release();
        global_nel.release();
        global_rho.release();

        return timer.getTime();

      };    
  }
}

