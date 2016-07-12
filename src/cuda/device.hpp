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


#ifndef DEVICE_HPP_
#define DEVICE_HPP_


#include "utils.hpp" 
#include "internal.h"
#include <boost/math/special_functions/digamma.hpp>

//Change these values depending on the CUDA architecture of your GPU
#define MAX_THREADS_PER_BLOCK 1024
#define MIN_BLOCKS_PER_SM 2
#define MAX_BLOCKS_PER_SM 32
#define MAX_THREADS_PER_SM 2048  
 
    
//This is wrong: __CUDA_ARCH__ is only defined in device
//Use the commented lines as an aid to set up the global vars defined above
//#ifndef MAX_THREADS_PER_BLOCK
//#if __CUDA_ARCH__ >= 500
    //#define MAX_THREADS_PER_BLOCK 1024
    //#define MIN_BLOCKS_PER_SM 2
    //#define MAX_BLOCKS_PER_SM 32
    //#define MAX_THREADS_PER_SM 2048    
//#elif __CUDA_ARCH__ >= 300
    //#define MAX_THREADS_PER_BLOCK 1024
    //#define MIN_BLOCKS_PER_SM 2
    //#define MAX_BLOCKS_PER_SM 16
    //#define MAX_THREADS_PER_SM 2048       
//#elif __CUDA_ARCH__>=200
    //#define MAX_THREADS_PER_BLOCK 512
    //#define MIN_BLOCKS_PER_SM 3
    //#define MAX_BLOCKS_PER_SM 8
    //#define MAX_THREADS_PER_SM 1536  
//#else  (Note: program is probably not working with GPU with arch < 200)
    //#define MAX_THREADS_PER_BLOCK 512
    //#define MIN_BLOCKS_PER_SM 2
    //#define MAX_BLOCKS_PER_SM 8
    //#define MAX_THREADS_PER_SM 1024 
//#endif

#define MIN_OPTIMAL_THREADS_PER_BLOCK MAX_THREADS_PER_SM/MAX_BLOCKS_PER_SM
//#endif

        

namespace RGBID_SLAM
{
  namespace device
  {    
    __device__ __forceinline__ float3
    operator* (const Mat33& m, const float3& vec)
    {
      return make_float3 (dot (m.data[0], vec), dot (m.data[1], vec), dot (m.data[2], vec));
    }
    
    inline float
    digamma_func(float x)
    {
      return boost::math::digamma(x);
    }
      
    
    class cudaTimer
    {
      public:
        inline cudaTimer()
        {
          cudaEventCreate(&start_);
          cudaEventRecord(start_,0);
        }
        
        inline float getTime()
        {
          float elapsed_time; 
          cudaEventCreate(&stop_); 
          cudaEventRecord(stop_,0); 
          cudaEventSynchronize(stop_); 
          cudaEventElapsedTime(&elapsed_time, start_,stop_); 
          
          return elapsed_time;
        }
      
      private:  
        cudaEvent_t start_;
        cudaEvent_t stop_;
    };
    
    
    inline void findBest2DBlockGridCombination(int gridSize, int blockSizeX, int blockSizeY, int cols, int rows, int& gridX, int& gridY)
    {
      int maxGridX = min(gridSize, divUp(cols,blockSizeX) );
      int maxGridY = min(gridSize, divUp(rows,blockSizeY) ); 
      int minPixelsPerThread = cols*rows;
      
      for (int newGridX = 1; newGridX < maxGridX; newGridX++)
      {
        if (newGridX > (maxGridX / 2))
          newGridX = maxGridX;
          
        int newGridY = gridSize / newGridX;
        
        if (newGridY > maxGridY)
          continue;
        
        int PixelColsPerThread = divUp(cols,newGridX*blockSizeX);
        int PixelRowsPerThread = divUp(rows,newGridY*blockSizeY);
        int pixelsPerThread = PixelColsPerThread*PixelRowsPerThread;
        
        if (pixelsPerThread < minPixelsPerThread) 
        {
          minPixelsPerThread = pixelsPerThread;
          gridX = newGridX;
          gridY = newGridY;
          
          if ( ( cols % (gridX*blockSizeX) == 0) && ( rows % (gridY*blockSizeY) == 0) && (gridX*gridY==gridSize) )
            return;
        }
      }  
      
      for (int newGridY = 1; newGridY < maxGridY; newGridY++)
      {
        if (newGridY > (maxGridY / 2))
          newGridY = maxGridY;
          
        int newGridX = gridSize / newGridY;
        
        if (newGridX > maxGridX)
          continue;
        
        int PixelColsPerThread = divUp(cols,newGridX*blockSizeX);
        int PixelRowsPerThread = divUp(rows,newGridY*blockSizeY);
        int pixelsPerThread = PixelColsPerThread*PixelRowsPerThread;
        
        if (pixelsPerThread < minPixelsPerThread)
        {
          minPixelsPerThread = pixelsPerThread;
          gridX = newGridX;
          gridY = newGridY;
          
          if ( ( cols % (gridX*blockSizeX) == 0) && ( rows % (gridY*blockSizeY) == 0) && (gridX*gridY==gridSize) )
            return;
        }
      }        
    };
    
    inline void computeBlockDim2D(int& gridX, int& gridY, int dimX, int dimY, int cols, int rows, int numSMs = -1)
    {
      int nThreadsPerBlock = dimX*dimY;
      int numMaxThreadsPerSM, numMaxBlocks, numMaxBlocksPerSM;
      
      if ((numSMs < 1) || (numSMs > dev_prop.multiProcessorCount))
      {
        numSMs = dev_prop.multiProcessorCount;
      }
      
      numMaxThreadsPerSM = dev_prop.maxThreadsPerMultiProcessor;
      
      numMaxBlocksPerSM = std::min(MAX_BLOCKS_PER_SM, divUp(numMaxThreadsPerSM + 1, nThreadsPerBlock) - 1); //Should be always equal, but jut in case
      numMaxBlocks = numSMs*numMaxBlocksPerSM;    
     
      int nBlocksInit = gridX*gridY;
      
      if (nBlocksInit > numMaxBlocks)
      {                    
        findBest2DBlockGridCombination(numMaxBlocks, dimX, dimY, cols, rows, gridX, gridY);                                             
      }   
    };
    
    
    inline void computeBlockDim1D(int& grid, int dim, int numSMs = -1)
    {
      int nThreadsPerBlock = dim;
      int numMaxThreadsPerSM, numMaxBlocks, numMaxBlocksPerSM;
      
      if ((numSMs < 1) || (numSMs > dev_prop.multiProcessorCount))
      {
        numSMs = dev_prop.multiProcessorCount;
      }
      
      numMaxThreadsPerSM = dev_prop.maxThreadsPerMultiProcessor;
      
      numMaxBlocksPerSM = std::min(MAX_BLOCKS_PER_SM, divUp(numMaxThreadsPerSM + 1, nThreadsPerBlock) - 1); //Should be always equal, but jut in case
      numMaxBlocks = numSMs*numMaxBlocksPerSM;   
     
      grid = min(grid,numMaxBlocks);
    };
    
  }
}


#endif /* DEVICE_HPP_*/
