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

#ifndef CLOUDSEGMENTER_HPP_
#define CLOUDSEGMENTER_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <set>

#include "types.h"
#include "settings.h"
#include "graph_segmenter.h"


namespace RGBID_SLAM 
{  
  struct Superjut
  {
    Superjut(){};
    ~Superjut()
    {        
      cloud_indices_.clear();
      normal_hist_.clear();
    }; 
    
    std::set<int>  cloud_indices_;
    PixelRGB color_;
    std::vector< std::pair<float, Point3D> > normal_hist_;
    float normal_entropy_;
    
    void
    computeHistogramOfNormalsAndEntropy(pcl::PointCloud<pcl::Normal>::Ptr normal_cloud, int Nbins);
  };
  
  class CloudSegmenter
  {
    public:
    
      CloudSegmenter(float kth = 0.6f, int min_segment_size = 300);
      
      void loadSettings(const Settings& settings);  
      
      void uploadPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud, pcl::PointCloud<pcl::Normal>::Ptr normal_cloud);
            
      void computeEdgesPixelNeighbors(const std::vector<int> &cloud2image_indices, const std::vector<int> &image2cloud_indices, 
                                                          int cols, int rows);
      
      void doSegmentation(std::vector<Superjut>& superjuts_list);
      
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_;
      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_;
      
    private:
      
      float computeCurvature (pcl::PointXYZRGB &p1, pcl::PointXYZRGB &p2, pcl::Normal &n1, pcl::Normal &n2);
      std::vector<Segmentation::Edge > edges_;
    
      float kTh_;
      int min_segment_size_;
      
  };  
}

#endif /* CLOUDSEGMENTER_HPP_ */
