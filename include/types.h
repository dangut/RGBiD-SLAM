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

#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <deque>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//#include <Eigen/StdDeque>
//#include <Eigen/StdVector>

#include "../ThirdParty/pcl_gpu_containers/include/device_array.h"

using namespace pcl::gpu;
    
namespace RGBID_SLAM
{  
  namespace PnPweighting
  {
    enum{NO_WEIGHTS,DIST_WEIGHTS};
  };
  
  class Keyframe;
  typedef boost::shared_ptr<Keyframe> KeyframePtr;
  
  class Pose;
  typedef boost::shared_ptr<Pose> PosePtr;
  
  class PoseConstraint;
  typedef boost::shared_ptr<PoseConstraint> PoseConstraintPtr;
  
  class VisodoTracker;
  typedef boost::shared_ptr<VisodoTracker> VisodoTrackerPtr;
  
  class KeyframeManager;
  typedef boost::shared_ptr<KeyframeManager> KeyframeManagerPtr;
  
  class PoseGraph;
  typedef boost::shared_ptr<PoseGraph> PoseGraphPtr;
  
  class VisualizationManager;
  typedef boost::shared_ptr<VisualizationManager> VisualizationManagerPtr;
  
  class LoopCloser;
  typedef boost::shared_ptr<LoopCloser> LoopCloserPtr;
  
  class FeatureExtractor;
  typedef boost::shared_ptr<FeatureExtractor> FeatureExtractorPtr;
  
  class CloudSegmenter;
  typedef boost::shared_ptr<CloudSegmenter> CloudSegmenterPtr;

  //class PnPGraphOptimiser;
  //typedef boost::shared_ptr<PnPGraphOptimiser> PnPGraphOptimiserPtr;    
  
  class KeyframeAlign;
  typedef boost::shared_ptr<KeyframeAlign> KeyframeAlignPtr;   
  
  typedef std::map<int,Eigen::Affine3d,std::less<int>, Eigen::aligned_allocator<std::pair<const int, Eigen::Affine3d> > > IdToPoseMap;
  typedef std::map<int,pcl::PointCloud<pcl::PointXYZRGB>::Ptr > IdToPointCloudMap;
  
  struct PixelRGB
  {
    unsigned char r, g, b;
  };    
  
  struct Point2D
  {
    float x, y;
  };  
  
  template<class T>
  struct ImageWrapper
  {
    ImageWrapper():elem_size(sizeof(T))
    {};
    
    T* data;
    unsigned int cols;
    unsigned int rows;
    
    unsigned int step;
    unsigned int elemSize() const {return elem_size;}
    
    private:
      unsigned int elem_size;    
  };
  
  template<class T> 
  class BufferWrapper
  {
    private:
    
      std::deque<T> buffer_;
      boost::mutex m_;
      int max_size_;
      boost::condition_variable c_empty_;
      boost::condition_variable c_full_;
    
    public:
    
      void initialise(int max_size=1000)
      {
        buffer_.clear();
        max_size_ = max_size;
      }
      
      void push (T &last)
      {
        //pcl::ScopeTime t1 ("push");
        {
          boost::mutex::scoped_lock lock(m_);
          
          while (buffer_.size() == max_size_)
          {
            c_full_.wait(lock);
            std::cout << "Buffer is full!!" << std::endl;
          }
            
          buffer_.push_back(last);
        }
        
        c_empty_.notify_one();
      }
      
      void pop (T &first)
      {
        //pcl::ScopeTime t1 ("pop");
        {
          boost::mutex::scoped_lock lock(m_);
          
          while (buffer_.size() == 0)
            c_empty_.wait(lock);
            
          first = buffer_.front();
            
          buffer_.pop_front();
        }
        
        c_full_.notify_one();
      }
      
      bool try_push (T &last)
      {
        //pcl::ScopeTime t1 ("push");
        {
          boost::mutex::scoped_lock lock(m_);
          
          if (buffer_.size() == max_size_)
          {
            std::cout << "Buffer is full!!" << std::endl;
            return false;
          }
            
          buffer_.push_back(last);
        }
        
        c_empty_.notify_one();
        
        return true;
      }
      
      bool try_pop (T &first)
      {        
        //pcl::ScopeTime t1 ("pop");
        {
          boost::mutex::scoped_lock lock(m_);
          
          if (buffer_.size() == 0)
            return false;
            
          first = buffer_.front();
            
          buffer_.pop_front();
        }
        
        c_full_.notify_one();
        
        return true;
      }
      
      void pop_all (std::vector<T> &popped_list)
      {
        {
          boost::mutex::scoped_lock lock(m_);            
          
          while (buffer_.size() == 0)
            c_empty_.wait(lock);              
          
          popped_list.clear();
          popped_list.reserve(buffer_.size());
          
          while (buffer_.size() != 0)         
          {
            popped_list.push_back(buffer_.front());
            buffer_.pop_front();            
          }
        }
        
        c_full_.notify_one();
      }
      
      void push_all (std::vector<T> &pushing_list)
      {
        {
          boost::mutex::scoped_lock lock(m_);
          
          while (buffer_.size() >= (max_size_-pushing_list.size()))
          {
            c_full_.wait(lock);
            std::cout << "Buffer is full!!" << std::endl;
          }
          
          for (int i=0; i<pushing_list.size(); i++)         
          {
            buffer_.push_back(pushing_list[i]);          
          }
        }
        
        c_empty_.notify_one();
      }
      
      bool try_pop_all (std::vector<T> &popped_list)
      {
        {
          boost::mutex::scoped_lock lock(m_);            
          
          while (buffer_.size() == 0)
            return false;              
          
          popped_list.clear();
          popped_list.reserve(buffer_.size());
          
          while (buffer_.size() != 0)         
          {
            popped_list.push_back(buffer_.front());
            buffer_.pop_front();            
          }
        }
        
        c_full_.notify_one();
        
        return true;
      }
      
      bool try_push_all (std::vector<T> &pushing_list)
      {
        {
          boost::mutex::scoped_lock lock(m_);
          
          if (buffer_.size() >= (max_size_-pushing_list.size()))
          {              
            std::cout << "Buffer is full!!" << std::endl;
            return false;
          }
          
          for (int i=0; i<pushing_list.size(); i++)         
          {
            buffer_.push_back(pushing_list[i]);          
          }
        }
        
        c_empty_.notify_one();
        
        return true;
      }
  };	 
  
  
  template<class T> 
  class BufferWrapperEigen
  {
    private:
    
      std::deque<T, Eigen::aligned_allocator<T> > buffer_;
      boost::mutex m_;
      int max_size_;
      boost::condition_variable c_empty_;
      boost::condition_variable c_full_;
    
    public:
    
      void initialise(int max_size=1000)
      {
        buffer_.clear();
        max_size_ = max_size;
      }
      
      void push (T &last)
      {
        //pcl::ScopeTime t1 ("push");
        {
          boost::mutex::scoped_lock lock(m_);
          
          while (buffer_.size() == max_size_)
          {
            c_full_.wait(lock);
            std::cout << "Buffer is full!!" << std::endl;
          }
            
          buffer_.push_back(last);
        }
        
        c_empty_.notify_one();
      }
      
      void pop (T &first)
      {
        //pcl::ScopeTime t1 ("pop");
        {
          boost::mutex::scoped_lock lock(m_);
          
          while (buffer_.size() == 0)
            c_empty_.wait(lock);
            
          first = buffer_.front();
            
          buffer_.pop_front();
        }
        
        c_full_.notify_one();
      }
      
      bool try_push (T &last)
      {
        //pcl::ScopeTime t1 ("push");
        {
          boost::mutex::scoped_lock lock(m_);
          
          if (buffer_.size() == max_size_)
          {
            std::cout << "Buffer is full!!" << std::endl;
            return false;
          }
            
          buffer_.push_back(last);
        }
        
        c_empty_.notify_one();
        
        return true;
      }
      
      bool try_pop (T &first)
      {
        
        //pcl::ScopeTime t1 ("pop");
        {
          boost::mutex::scoped_lock lock(m_);
          
          if (buffer_.size() == 0)
            return false;
            
          first = buffer_.front();
            
          buffer_.pop_front();
        }
        
        c_full_.notify_one();
        
        return true;
      }
      
      void pop_all (std::vector<T, Eigen::aligned_allocator<T> > &popped_list)
      {
        {
          boost::mutex::scoped_lock lock(m_);            
          
          while (buffer_.size() == 0)
            c_empty_.wait(lock);              
          
          popped_list.clear();
          popped_list.reserve(buffer_.size());
          
          while (buffer_.size() != 0)         
          {
            popped_list.push_back(buffer_.front());
            buffer_.pop_front();            
          }
        }
        
        c_full_.notify_one();
      }
      
      void push_all (std::vector<T, Eigen::aligned_allocator<T> > &pushing_list)
      {
        {
          boost::mutex::scoped_lock lock(m_);
          
          while (buffer_.size() >= (max_size_-pushing_list.size()))
          {
            c_full_.wait(lock);
            std::cout << "Buffer is full!!" << std::endl;
          }
          
          for (int i=0; i<pushing_list.size(); i++)         
          {
            buffer_.push_back(pushing_list[i]);          
          }
        }
        
        c_empty_.notify_one();
      }
      
      bool try_pop_all (std::vector<T, Eigen::aligned_allocator<T> > &popped_list)
      {
        {
          boost::mutex::scoped_lock lock(m_);            
          
          while (buffer_.size() == 0)
            return false;              
          
          popped_list.clear();
          popped_list.reserve(buffer_.size());
          
          while (buffer_.size() != 0)         
          {
            popped_list.push_back(buffer_.front());
            buffer_.pop_front();            
          }
        }
        
        c_full_.notify_one();
        
        return true;
      }
      
      bool try_push_all (std::vector<T, Eigen::aligned_allocator<T> > &pushing_list)
      {
        {
          boost::mutex::scoped_lock lock(m_);
          
          if (buffer_.size() >= (max_size_-pushing_list.size()))
          {              
            std::cout << "Buffer is full!!" << std::endl;
            return false;
          }
          
          for (int i=0; i<pushing_list.size(); i++)         
          {
            buffer_.push_back(pushing_list[i]);          
          }
        }
        
        c_empty_.notify_one();
        
        return true;
      }
  };	   

  //typedef visodoRGBD::gpu::PixelRGB PixelRGB;
  typedef DeviceArray2D<PixelRGB> View;
  typedef DeviceArray2D<unsigned short> DepthMap;
  typedef DeviceArray2D<unsigned char> IntensityMap;
  typedef DeviceArray2D<float> DepthMapf;
  typedef DeviceArray2D<float> IntensityMapf;
  typedef DeviceArray2D<float> GradientMap;
  typedef DeviceArray2D<float> MapArr; 
  typedef DeviceArray2D<unsigned char> BinaryMap;  
  
  typedef Eigen::Matrix<float,6,6> Matrix6f; 
  typedef Eigen::Matrix<float,6,1> Vector6f;
  
  typedef double float_trafos;          

  typedef Eigen::Matrix<float_trafos, 3, 3, Eigen::RowMajor> Matrix3ft;
  typedef Eigen::Matrix<float_trafos,3,1> Vector3ft;
  typedef Eigen::Matrix<float_trafos,6,1> Vector6ft;
  typedef Eigen::Matrix<float_trafos, 4, 4, Eigen::RowMajor> Matrix4ft;
  typedef Eigen::Matrix<float_trafos, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXft;
  
  typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3f;
  typedef Eigen::Matrix<float,3,1> Vector3f;
  typedef Eigen::Matrix<float,6,1> Vector6f;
  typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4f;
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf;
  
     
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  
  typedef boost::shared_ptr< DeviceArray2D<float> > DeviceArray2DPtr;
  
  struct Point3D
  {
    float x;
    float y;
    float z;
  };   
  
}
#endif /* TYPES_HPP_ */
