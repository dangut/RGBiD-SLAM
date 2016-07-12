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

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include "keyframe_manager.h"
#include "cloud_segmenter.h"
#include "util_funcs.h"


void
RGBID_SLAM::Superjut::computeHistogramOfNormalsAndEntropy(pcl::PointCloud<pcl::Normal>::Ptr normal_cloud, int Nbins)
{
  std::vector<Point3D> bin_centers;
  goldenSectionSpiral (bin_centers, Nbins); 
  
  normal_hist_.resize(Nbins);
  
  for (int j=0; j<Nbins; j++)
  {
    normal_hist_[j].first = 0.f;
    normal_hist_[j].second = bin_centers[j];
  }
  
  int jut_size = cloud_indices_.size();
  
  for (std::set<int>::const_iterator it_point_id = cloud_indices_.begin();
            it_point_id != cloud_indices_.end(); it_point_id++)
  {
    pcl::Normal &ni = normal_cloud->points[(*it_point_id)];
    
    float maxdotprod = -1.1f;
    int jmax = -1;
    //I assume that pcl::Normal is yet normalized
    
    for (int j=0; j<Nbins; j++)
    {
      float dotprod = ni.normal_x*bin_centers[j].x +
                      ni.normal_y*bin_centers[j].y +
                      ni.normal_z*bin_centers[j].z;
                      
      if (dotprod > maxdotprod)
      {
        jmax = j;
        maxdotprod = dotprod;
      }
    }
    
    normal_hist_[jmax].first += (1.f / (float) jut_size);    
  } 
  
  normal_entropy_= 0.f;
  
  for (int j=0; j<Nbins; j++)
  {
		float freq = normal_hist_[j].first;
		normal_entropy_ += ( ( freq < ( 1.f / (float) (2*jut_size) ) ) ? 0.f : -freq*log( freq ) );
		//std::cout << "frequency: " << freq << std::endl;
	}
  normal_entropy_ /= log((float) jut_size);
}


RGBID_SLAM::CloudSegmenter::CloudSegmenter(float kTh, int min_segment_size)
{
  kTh_ = kTh;
  min_segment_size_ = min_segment_size;
  point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>) ;
  normal_cloud_.reset(new pcl::PointCloud<pcl::Normal>) ;
}  

void 
RGBID_SLAM::CloudSegmenter::loadSettings(const Settings& settings)
{
  Section cloud_segmenter_section;
  
  if (settings.getSection("CLOUD_SEGMENTER",cloud_segmenter_section))
  {
    std::cout << "CLOUD_SEGMENTER" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    
    Entry entry;
    
    if (cloud_segmenter_section.getEntry("K_THRESHOLD",entry))
    {
      std::string kTh_str = entry.getValue();
      
      kTh_ = atof(kTh_str.c_str());
      
      std::cout << "  K_THRESHOLD: " << kTh_ << std::endl;    
    }
    
    if (cloud_segmenter_section.getEntry("MIN_SEGMENT_SIZE",entry))
    {
      std::string min_segment_size_str = entry.getValue();
      
      min_segment_size_ = atoi(min_segment_size_str.c_str());
      
      std::cout << "  MIN_SEGMENT_SIZE: " << min_segment_size_ << std::endl;    
    }
    
    std::cout << std::endl;
    std::cout << std::endl;
  }  
}  

void
RGBID_SLAM::CloudSegmenter::uploadPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud, pcl::PointCloud<pcl::Normal>::Ptr normal_cloud)
{
  copyPointCloud(point_cloud, point_cloud_);
  copyNormalCloud(normal_cloud, normal_cloud_);    
}

void
RGBID_SLAM::CloudSegmenter::computeEdgesPixelNeighbors(const std::vector<int> &cloud2image_indices, const std::vector<int> &image2cloud_indices, 
                                                            int cols, int rows)
{
  edges_.clear();
  
  for (int i = 0; i < cloud2image_indices.size(); i++)
  {
    int x = cloud2image_indices[i] % cols;
    int y = cloud2image_indices[i] / cols;
    
    int x_nb;
    int y_nb;
    
    for (int nb = 0; nb < 4; nb++)
    {
      switch (nb)
      {
        case 0:
          x_nb = x+1;
          y_nb = y;
          break;
        case 1:
          x_nb = x;
          y_nb = y+1;
          break;
        case 2:
          x_nb = x+1;
          y_nb = y-1;
          break;
        case 3:
          x_nb = x+1;
          y_nb = y+1;
          break;
        case 4:
          x_nb = x-1;
          y_nb = y+1;
          break;
        case 5:
          x_nb = x-1;
          y_nb = y;
          break;
        case 6:
          x_nb = x-1;
          y_nb = y-1;
          break;
        case 7:
          x_nb = x;
          y_nb = y-1;
          break;
        default:
          break;
      }
    
    
      if ( (x_nb >= 0) && (x_nb < cols) && (y_nb >=0 ) && (y_nb < rows) )
      {
        int im_index = y_nb*cols + x_nb;
        int i_nb = image2cloud_indices.at(im_index);
        
        if (!(i_nb < 0))
        {          
          pcl::PointXYZRGB &p1= point_cloud_->points[i];
          pcl::PointXYZRGB &p2= point_cloud_->points[i_nb];
          pcl::Normal &n1= normal_cloud_->points[i];
          pcl::Normal &n2= normal_cloud_->points[i_nb];
          
          Segmentation::Edge e;
          e.ini_ = i;
          e.end_ = i_nb;
          e.w_ = computeCurvature(p1,p2,n1,n2);
          
          if (!(std::isnan(e.w_)))
            edges_.push_back(e);
                   
        }
      }
    }
  }
}

float
RGBID_SLAM::CloudSegmenter::computeCurvature (pcl::PointXYZRGB &p1, pcl::PointXYZRGB &p2, pcl::Normal &n1, pcl::Normal &n2)
{
  float dx= p2.x-p1.x;
  float dy= p2.y-p1.y;
  float dz= p2.z-p1.z;
  float norm_dp= sqrt(dx*dx+dy*dy+dz*dz); 
  float dot= n1.normal_x*n2.normal_x + n1.normal_y*n2.normal_y + n1.normal_z*n2.normal_z;
  float dot2= (1.f/norm_dp)*(n2.normal_x*dx + n2.normal_y*dy + n2.normal_z*dz);
  float c= 1.0 - dot;
  
  if (dot2>0)
    c = c*c;
  
  float dist = std::sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
           
  //if (dist > 0.05)  
    //c = std::numeric_limits<float>::quiet_NaN();
  
  return  c;
}

  
  
void 
RGBID_SLAM::CloudSegmenter::doSegmentation(std::vector<Superjut>& superjuts_list)
{  
  srand(1); 
  
  int Npoints = point_cloud_->points.size();
  
  Segmentation::Graph segmentation_graph(kTh_, min_segment_size_, Npoints);
  
  superjuts_list.clear();

  /////////////////
  //Felzenswalb segmentation and segment fusion
  pcl::StopWatch tseg;
  
  segmentation_graph.applySegmentation(edges_);  
  
  std::vector<int> point2jut_indices(Npoints, -1);
  PixelRGB seg_color;

  for(int i=0; i<Npoints; i++) 
  {    
    int ip = segmentation_graph.findParent(i);
   
    if (point2jut_indices[ip] >= 0)
    {    
      // add it on top of the (already created) cloud for this segment
      int jut_id = point2jut_indices[ip];
      (superjuts_list[jut_id].cloud_indices_).insert(i);
    } 
    else 
    {      
      Superjut new_superjut;  
      new_superjut.cloud_indices_.insert(i);
      
      // add new color
      seg_color.r= (rand() % 205)+50;
      seg_color.g= (rand() % 205)+50;
      seg_color.b= (rand() % 205)+50;
      new_superjut.color_ = seg_color;

      superjuts_list.push_back(new_superjut);
      point2jut_indices[ip] = superjuts_list.size() - 1;     
    }
  }  
} 
  


