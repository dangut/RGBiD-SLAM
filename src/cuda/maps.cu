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
    __global__ void
    computeVmapKernel (const PtrStepSz<float> depth_inv, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy)
    {
      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      for (int u = tid_x;   u < depth_inv.cols; u += blockDim.x * gridDim.x)
      { 
        for (int v = tid_y;   v < depth_inv.rows; v += blockDim.y * gridDim.y)    
        {       
          
            float z = (1.f / depth_inv.ptr (v)[u]); 

            if (!isnan(z))
            {
              float vx = z * (u - cx) * fx_inv;
              float vy = z * (v - cy) * fy_inv;
              float vz = z;

              vmap.ptr (v                 )[u] = vx;
              vmap.ptr (v + depth_inv.rows    )[u] = vy;
              vmap.ptr (v + depth_inv.rows * 2)[u] = vz;
            }
            else
              vmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
        }
      }
    }

    __global__ void
    computeNmapKernel (int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
    {
      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      for (int u = tid_x;   u < cols; u += blockDim.x * gridDim.x)
      { 
        for (int v = tid_y;   v < rows; v += blockDim.y * gridDim.y)    
        {            
          nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();

          if (u == cols - 1 || v == rows - 1)
          {
            continue;
          }

          float3 v00, v01, v10;
          v00.x = vmap.ptr (v  )[u];
          v01.x = vmap.ptr (v  )[u + 1];
          v10.x = vmap.ptr (v + 1)[u];

          if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
          {
            v00.y = vmap.ptr (v + rows)[u];
            v01.y = vmap.ptr (v + rows)[u + 1];
            v10.y = vmap.ptr (v + 1 + rows)[u];

            v00.z = vmap.ptr (v + 2 * rows)[u];
            v01.z = vmap.ptr (v + 2 * rows)[u + 1];
            v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

            float3 r = normalized (cross (v01 - v00, v10 - v00));

            nmap.ptr (v       )[u] = r.x;
            nmap.ptr (v + rows)[u] = r.y;
            nmap.ptr (v + 2 * rows)[u] = r.z;
          }
        }
      }
    }
    
    __global__ void
    computeNmapGradientsKernel (int rows, int cols, const PtrStepSz<float> depth_inv, const PtrStepSz<float> grad_x, const PtrStepSz<float> grad_y, PtrStep<float> nmap, float fx, float fy, float cx, float cy)
    {
      int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
      
      for (int u = tid_x;   u < depth_inv.cols; u += blockDim.x * gridDim.x)
      { 
        for (int v = tid_y;   v < depth_inv.rows; v += blockDim.y * gridDim.y)    
        {       
          nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();            
          
          float w = depth_inv.ptr (v)[u]; 
          float gx = grad_x.ptr (v)[u]; 
          float gy = grad_y.ptr (v)[u];             
          
          if ( !(isnan(w) || isnan(gx) || isnan(gy) ))
          {
          
            float3 n_unnorm;
            n_unnorm.x = gx*fx;
            n_unnorm.y = gy*fy;
            n_unnorm.z = gx*(cx-u) + gy*(cy-v) + w;
            
            float3 n = normalized (n_unnorm);
            
            float3 vt;
            float z = 1.f/w;
            vt.x = z * (u - cx) * (1.f/fx);
            vt.y = z * (v - cy) * (1.f/fy);
            vt.z = z;
            
            float3 vt_norm = normalized(vt);
              
            float acos_vn = dot(vt_norm,n);
            
            if (acos_vn > 0.1)
            {     
              nmap.ptr (v       )[u] = n.x;
              nmap.ptr (v + rows)[u] = n.y;
              nmap.ptr (v + 2 * rows)[u] = n.z;
            }                
          }
        }
      }
    }

  
  enum
      {
        kx = 3,
        ky = 3,
        STEP = 1
      };

      __global__ void
      computeNmapKernelEigen (int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
      {
        int tid_x = blockIdx.x * blockDim.x + threadIdx.x;  
        int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
        
        for (int u = tid_x;   u < cols; u += blockDim.x * gridDim.x)
        { 
          for (int v = tid_y;   v < rows; v += blockDim.y * gridDim.y)    
          {   

            nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();

            if (!(isnan (vmap.ptr (v)[u])))
            {
              int ty = min (v - ky / 2 + ky, rows - 1);
              int tx = min (u - kx / 2 + kx, cols - 1);

              float3 centroid = make_float3 (0.f, 0.f, 0.f);
              int counter = 0;
              for (int cy = max (v - ky / 2, 0); cy < ty; cy += STEP)
                for (int cx = max (u - kx / 2, 0); cx < tx; cx += STEP)
                {
                  float v_x = vmap.ptr (cy)[cx];
                  if (!isnan (v_x))
                  {
                    centroid.x += v_x;
                    centroid.y += vmap.ptr (cy + rows)[cx];
                    centroid.z += vmap.ptr (cy + 2 * rows)[cx];
                    ++counter;
                  }
                }

              //int side = (kx-1 / STEP) + 1;
              //int area = side*side;
              
              if (counter >= 4)
              {
                centroid *= 1.f / counter;

                float cov[] = {0, 0, 0, 0, 0, 0};

                for (int cy = max (v - ky / 2, 0); cy < ty; cy += STEP)
                  for (int cx = max (u - kx / 2, 0); cx < tx; cx += STEP)
                  {
                    float3 vt;
                    vt.x = vmap.ptr (cy)[cx];
                    if (isnan (vt.x))
                      continue;

                    vt.y = vmap.ptr (cy + rows)[cx];
                    vt.z = vmap.ptr (cy + 2 * rows)[cx];

                    float3 d = vt - centroid;

                    cov[0] += d.x * d.x;               //cov (0, 0)
                    cov[1] += d.x * d.y;               //cov (0, 1)
                    cov[2] += d.x * d.z;               //cov (0, 2)
                    cov[3] += d.y * d.y;               //cov (1, 1)
                    cov[4] += d.y * d.z;               //cov (1, 2)
                    cov[5] += d.z * d.z;               //cov (2, 2)
                  }

                typedef Eigen33::Mat33 Mat33;
                Eigen33 eigen33 (cov);

                Mat33 tmp;
                Mat33 vec_tmp;
                Mat33 evecs;
                float3 evals;
                eigen33.compute (tmp, vec_tmp, evecs, evals);

                float3 n = normalized (evecs[0]);

                //u = threadIdx.x + blockIdx.x * blockDim.x;
                //v = threadIdx.y + blockIdx.y * blockDim.y;
                
                float3 vt;
                vt.x = vmap.ptr (v)[u];
                if (isnan (vt.x))
                  continue;

                vt.y = vmap.ptr (v + rows)[u];
                vt.z = vmap.ptr (v + 2 * rows)[u];
                
                float3 vt_norm = normalized(vt);
                
                float acos_vn = dot(vt_norm,n);
                
                if (acos_vn > 0.1)
                {     
                  nmap.ptr (v       )[u] = n.x;
                  nmap.ptr (v + rows)[u] = n.y;
                  nmap.ptr (v + 2 * rows)[u] = n.z;
                }                
                
              }
            }
          }
        }
      }
   }
}



namespace RGBID_SLAM
{
  namespace device
  {
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    createVMap (const Intr& intr, const DepthMapf& depth_inv, MapArr& vmap, int numSMs)
    {
      vmap.create (depth_inv.rows () * 3, depth_inv.cols ());
            
      dim3 block (32, 8);
      
      int gridX = divUp (depth_inv.cols (), block.x);
      int gridY = divUp (depth_inv.rows (), block.y);
      
      ///////////////////////
      
      //std::cout << "adaptive_pyr" << std::endl;
      int nThreadsPerBlock = 32*8;
      int numMaxThreadsPerSM, numMaxBlocks, numMaxBlocksPerSM;
      
      if ((numSMs < 1) || (numSMs > dev_prop.multiProcessorCount))
      {
        numSMs = dev_prop.multiProcessorCount;
      }
      
      numMaxThreadsPerSM = dev_prop.maxThreadsPerMultiProcessor;
      
      numMaxBlocksPerSM = std::min(MAX_BLOCKS_PER_SM, divUp(numMaxThreadsPerSM + 1, nThreadsPerBlock) - 1); //Should be always equal, but jut in case
      numMaxBlocks = numSMs*numMaxBlocksPerSM;
      //numMaxBlocks = std::min(120, numMaxBlocks);
      int nBlocksInit = gridX*gridY;
      
      if (nBlocksInit > numMaxBlocks)
      {                    
        findBest2DBlockGridCombination(numMaxBlocks, 32, 8, depth_inv.cols (), depth_inv.rows (), gridX, gridY);                                             
      }   
      //std::cout << "pyrdown: " << gridX << " " << gridY << std::endl;  
      
      //////////////////
      
      dim3 grid (gridX, gridY);

      float fx = intr.fx, cx = intr.cx;
      float fy = intr.fy, cy = intr.cy;

      computeVmapKernel<<<grid, block>>>(depth_inv, vmap, 1.f / fx, 1.f / fy, cx, cy);
      cudaSafeCall (cudaGetLastError ());
      cudaSafeCall (cudaStreamSynchronize(0)); 
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    createNMap (const MapArr& vmap, MapArr& nmap, int numSMs)
    {
      nmap.create (vmap.rows (), vmap.cols ());

      int rows = vmap.rows () / 3;
      int cols = vmap.cols ();

      dim3 block (32, 8);
      
      int gridX = divUp (cols, block.x);
      int gridY = divUp (rows, block.y);
      
      ///////////////////////
      
      //std::cout << "adaptive_pyr" << std::endl;
      int nThreadsPerBlock = 32*8;
      int numMaxThreadsPerSM, numMaxBlocks, numMaxBlocksPerSM;
      
      if ((numSMs < 1) || (numSMs > dev_prop.multiProcessorCount))
      {
        numSMs = dev_prop.multiProcessorCount;
      }
      
      numMaxThreadsPerSM = dev_prop.maxThreadsPerMultiProcessor;
      
      numMaxBlocksPerSM = std::min(MAX_BLOCKS_PER_SM, divUp(numMaxThreadsPerSM + 1, nThreadsPerBlock) - 1); //Should be always equal, but jut in case
      numMaxBlocks = numSMs*numMaxBlocksPerSM;
      //numMaxBlocks = std::min(120, numMaxBlocks);
      int nBlocksInit = gridX*gridY;
      
      if (nBlocksInit > numMaxBlocks)
      {                    
        findBest2DBlockGridCombination(numMaxBlocks, 32, 8, cols, rows, gridX, gridY);                                             
      }   
      //std::cout << "pyrdown: " << gridX << " " << gridY << std::endl;  
      
      //////////////////
      
      dim3 grid (gridX, gridY);

      computeNmapKernelEigen<<<grid, block>>>(rows, cols, vmap, nmap);
      //computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
      cudaSafeCall (cudaGetLastError ());
      cudaSafeCall (cudaStreamSynchronize(0)); 
    }
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    createNMapGradients (const Intr& intr, const DepthMapf& depth_inv, const GradientMap& grad_x, const GradientMap& grad_y, MapArr& nmap, int numSMs)
    {
      nmap.create (depth_inv.rows () * 3, depth_inv.cols ());

      int rows = depth_inv.rows ();
      int cols = depth_inv.cols ();

      dim3 block (32, 8);
      
      int gridX = divUp (cols, block.x);
      int gridY = divUp (rows, block.y);
      
      ///////////////////////
      
      //std::cout << "adaptive_pyr" << std::endl;
      int nThreadsPerBlock = 32*8;
      int numMaxThreadsPerSM, numMaxBlocks, numMaxBlocksPerSM;
      
      if ((numSMs < 1) || (numSMs > dev_prop.multiProcessorCount))
      {
        numSMs = dev_prop.multiProcessorCount;
      }
      
      numMaxThreadsPerSM = dev_prop.maxThreadsPerMultiProcessor;
      
      numMaxBlocksPerSM = std::min(MAX_BLOCKS_PER_SM, divUp(numMaxThreadsPerSM + 1, nThreadsPerBlock) - 1); //Should be always equal, but jut in case
      numMaxBlocks = numSMs*numMaxBlocksPerSM;
      //numMaxBlocks = std::min(120, numMaxBlocks);
      int nBlocksInit = gridX*gridY;
      
      if (nBlocksInit > numMaxBlocks)
      {                    
        findBest2DBlockGridCombination(numMaxBlocks, 32, 8, cols, rows, gridX, gridY);                                             
      }   
      //std::cout << "pyrdown: " << gridX << " " << gridY << std::endl;  
      
      //////////////////
      
      dim3 grid (gridX, gridY);
      
      float fx = intr.fx, cx = intr.cx;
      float fy = intr.fy, cy = intr.cy;

      computeNmapGradientsKernel<<<grid, block>>>(rows, cols, depth_inv, grad_x, grad_y, nmap, fx, fy, cx, cy);
      cudaSafeCall (cudaGetLastError ());
      cudaSafeCall (cudaStreamSynchronize(0)); 
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*void
    createNMapEigen (const MapArr& vmap, MapArr& nmap)
    {
      int cols = vmap.cols ();
      int rows = vmap.rows () / 3;

      nmap.create (vmap.rows (), vmap.cols ());

      dim3 block (32, 8);
      dim3 grid (1, 1, 1);
      grid.x = divUp (cols, block.x);
      grid.y = divUp (rows, block.y);

      computeNmapKernelEigen<<<grid, block>>>(rows, cols, vmap, nmap);
      cudaSafeCall (cudaGetLastError ());
      cudaSafeCall (cudaStreamSynchronize(0));
    }
    */
  }
}







