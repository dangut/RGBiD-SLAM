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

#ifndef UTILFUNCS_HPP_
#define UTILFUNCS_HPP_

#include <Eigen/Core>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "../ThirdParty/pcl_gpu_containers/include/device_array.h"
#include "types.h"
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <sys/ioctl.h>
#include <linux/kd.h>

namespace RGBID_SLAM 
{
         
    Vector6ft logMap(const Matrix3ft& R, const Vector3ft& trans); 

    Matrix4ft expMap(const Vector3ft& omega_t, const Vector3ft& v_t);  
    
    Matrix3ft expMapRot(const Vector3ft& omega_t);
    
    Matrix3ft forceOrthogonalisation(const Matrix3ft& matrix);

    inline Eigen::Matrix3d skew(const Eigen::Vector3d& w)
    {
      Eigen::Matrix3d res;
      res <<  0.0, -w[2], w[1],
              w[2], 0.0, -w[0],
              -w[1], w[0], 0.0;       

      return res;
    }  
    
    void goldenSectionSpiral ( std::vector<Point3D>& spherePoints, int Npoints); 
    
    void computeBresenhamSemicircle(std::vector<int> &umax, int radius);
    
    bool compare_keypoints(const cv::KeyPoint& p1, const cv::KeyPoint& p2);
    
    void descVec2DescMat(std::vector<cv::Mat> &desc_vec, cv::Mat &desc_mat); 
    
    void alignPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud, Eigen::Matrix3d R_WC, Eigen::Vector3d t_WC);
    
    void 
    copyPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_dst);
    
    void 
    copyNormalCloud(pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_src, pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_dst);
    
    double
    computeConvexHullArea(const std::vector<cv::Point2f> &points);
    
    inline void
    playBeep()
    {
      int console_fd = -1;
      
      if((console_fd = open("/dev/tty0", O_WRONLY)) == -1)
        console_fd = open("/dev/vc/0", O_WRONLY);
        
      std::cout << "console_fd: " << console_fd << std::endl; 
      
      int ms = 1000;
      int freq = 440;
      ioctl(console_fd, KDMKTONE, (ms<<16 | 1193180/freq));
      
      close(console_fd);
    }
    
    template<class T>
    T computeWeightedMean(std::vector<T, Eigen::aligned_allocator<T> > samples, std::vector<double> weights)
    {
      T sum_samples = T::Zero();
      float sum_weights = 0.f;
      
      for (int i=0; i<samples.size(); i++)
      {
        sum_samples += weights[i]*samples[i];
        sum_weights += weights[i];
      }
      
      return (1.f / sum_weights)*sum_samples;
    };
    
    template<class Tvec, class Tmat>
    Tmat computeWeightedCorrelation(std::vector<Tvec, Eigen::aligned_allocator<Tvec> > samples1, 
                                    std::vector<Tvec, Eigen::aligned_allocator<Tvec> > samples2,
                                    std::vector<double> weights)
    {
      Tmat sum_cross_samples = Tmat::Zero();
      float sum_weights = 0.f;
      
      for (int i=0; i<samples1.size(); i++)
      {
        sum_cross_samples += weights[i]*(samples1[i]*samples2[i].transpose());
        sum_weights += weights[i];
      }
      
      return (1.f / sum_weights)*sum_cross_samples;
    };
  
}
#endif /* UTILFUNCS_HPP_ */
