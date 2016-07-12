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
 *  Copyright (c) 2011, Willow Garage, Inc.
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
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 *  Modified by Daniel Gutiérrez
 */

#include "evaluation.h"

#include <iostream>



//TUM KInect
const float Evaluation::fx = 525.0f;
const float Evaluation::fy = 525.0f;
const float Evaluation::cx = 319.5f;
const float Evaluation::cy = 239.5f;

//TUM KInect calib fr2
//const float Evaluation::fx = 520.9f;
//const float Evaluation::fy = 521.0f;
//const float Evaluation::cx = 325.1f;
//const float Evaluation::cy = 249.7f;

//TUM KInect fr3
//const float Evaluation::fx = 535.40f;
//const float Evaluation::fy = 539.2f;
//const float Evaluation::cx = 320.1f;
//const float Evaluation::cy = 247.6f;

////My asus (DAniel)
//const float Evaluation::fx = 543.78f;
//const float Evaluation::fy = 543.78f;
//const float Evaluation::cx = 313.45f;
//const float Evaluation::cy = 235.00f;

//Handa
//const float Evaluation::fx = 481.20f;
//const float Evaluation::fy = -480.0f;
//const float Evaluation::cx = 319.5f;
//const float Evaluation::cy = 239.5f;

#ifndef HAVE_OPENCV

struct Evaluation::Impl {};

Evaluation::Evaluation(const std::string&, const std::string& match_file) { cout << "Evaluation requires OpenCV. Please enable it in cmake-file" << endl; exit(0); }
void Evaluation::setMatchFile(const std::string&) { }
bool Evaluation::grab (double stamp, RGBID_SLAM::ImageWrapper<RGB>& rgb24) { return false; }
bool Evaluation::grab (double stamp, RGBID_SLAM::ImageWrapper<unsigned short>& depth) { return false; }
bool Evaluation::grab (int stamp, RGBID_SLAM::ImageWrapper<unsigned short>& depth, RGBID_SLAM::ImageWrapper<RGB>& rgb24) { return false; }
void Evaluation::saveAllPoses(const visodo::gpu::VisodoTracker& visodo, int frame_number, const std::string& logfile) const {}
void Evaluation::saveAllPoses(std::vector<RGBID_SLAM::Pose>& poses, const visodo::gpu::VisodoTracker& visodo, int frame_number, const std::string& logfile) const {}
void Evaluation::saveTimeLogFiles(const RGBID_SLAM::VisodoTracker& visodo, const RGBID_SLAM::KeyframeManager& kf_manager, const std::string& kftimes_logfile) const {}

#else

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<fstream>

//using namespace cv;

struct Evaluation::Impl
{
   cv::Mat depth_buffer;
   cv::Mat rgb_buffer;
};



Evaluation::Evaluation(const std::string& folder, const std::string& match_file) : folder_(folder), visualization_(false)
{   
  impl_.reset( new Impl() );

  if (folder_[folder_.size() - 1] != '\\' && folder_[folder_.size() - 1] != '/')
      folder_.push_back('/');

  if (!match_file.empty())
  {
      setMatchFile(match_file);
  }
  else
  {
    cout << "Initializing evaluation from folder: " << folder_ << endl;
    std::string depth_file = folder_ + "depth_associated.txt";
    std::string rgb_file = folder_ + "rgb_associated.txt";
    
    readFile(depth_file, depth_stamps_and_filenames_);
    readFile(rgb_file, rgb_stamps_and_filenames_);  

    std::cout << "Associate: " << folder_ << endl;
    associate_depth_rgb(depth_file, rgb_file);
  }

  //string associated_file = folder_ + "associated.txt";
}

void Evaluation::associate_depth_rgb(const std::string& file_depth, const std::string& file_rgb)
{
  char buffer[4096];

  std::string full_depth = file_depth;
  std::string full_rgb = file_rgb;

  std::ifstream iff_depth(full_depth.c_str());
  std::ifstream iff_rgb(full_rgb.c_str());

  if(!iff_depth || !iff_rgb)
  {
    std::cout << "Can't read rgbd" << file_depth << endl;
    exit(1);
  }
  // ignore three header lines
  iff_depth.getline(buffer, sizeof(buffer));
  iff_depth.getline(buffer, sizeof(buffer));
  iff_depth.getline(buffer, sizeof(buffer));

  // ignore three header lines
  iff_rgb.getline(buffer, sizeof(buffer));
  iff_rgb.getline(buffer, sizeof(buffer));
  iff_rgb.getline(buffer, sizeof(buffer));
  accociations_.clear();  
  while (!iff_depth.eof() || !iff_rgb.eof())
  {
    Association acc;    
    iff_depth >> acc.time1 >> acc.name1;
    iff_rgb  >> acc.time2 >> acc.name2;
    accociations_.push_back(acc);
  }    
}

void Evaluation::setMatchFile(const std::string& file)
{
  std::string full = folder_ + file;
  std::ifstream iff(full.c_str());  
  if(!iff)
  {
    std::cout << "Can't read " << file << endl;
    exit(1);
  }

  accociations_.clear();  
  while (!iff.eof())
  {
    Association acc;    
    iff >> acc.time1 >> acc.name1 >> acc.time2 >> acc.name2;
    accociations_.push_back(acc);
  }  
}

void Evaluation::readFile(const std::string& file, std::vector< std::pair<double,std::string> >& output)
{
  char buffer[4096];
  std::vector< std::pair<double,std::string> > tmp;
  
  std::ifstream iff(file.c_str());
  if(!iff)
  {
    std::cout << "Can't read" << file << endl;
    exit(1);
  }

  // ignore three header lines
  iff.getline(buffer, sizeof(buffer));
  iff.getline(buffer, sizeof(buffer));
  iff.getline(buffer, sizeof(buffer));
	
  // each line consists of the timestamp and the filename of the depth image
  while (!iff.eof())
  {
    double time; 
    std::string name;
    iff >> time >> name;
    tmp.push_back(std::make_pair(time, name));
  }
  tmp.swap(output);
}
  
bool Evaluation::grab (double stamp, RGBID_SLAM::ImageWrapper<RGB>& rgb24)
{  
  size_t i = static_cast<size_t>(stamp); // temporary solution, now it expects only index
  size_t total = accociations_.empty() ? rgb_stamps_and_filenames_.size() : accociations_.size();
  std::cout << "Grabbing" << endl;
  if ( i>= total)
      return false;
  
  std::string file = folder_ + (accociations_.empty() ? rgb_stamps_and_filenames_[i].second : accociations_[i].name2);

  cv::Mat bgr = cv::imread(file);
  if(bgr.empty())
      return false;     
      
  cv::cvtColor(bgr, impl_->rgb_buffer, CV_BGR2RGB);
  
  rgb24.data = impl_->rgb_buffer.ptr<RGB>();
  rgb24.cols = impl_->rgb_buffer.cols;
  rgb24.rows = impl_->rgb_buffer.rows;
  rgb24.step = impl_->rgb_buffer.cols*rgb24.elemSize();

  if (visualization_)
  {			    
	cv::imshow("Color channel", bgr);
	cv::waitKey(3);
  }
   std::cout << "End Grabbing" << file << endl;
  return true;  
}

bool Evaluation::grab (double stamp, RGBID_SLAM::ImageWrapper<unsigned short>& depth)
{  
  size_t i = static_cast<size_t>(stamp); // temporary solution, now it expects only index
  size_t total = accociations_.empty() ? depth_stamps_and_filenames_.size() : accociations_.size();

  if ( i>= total)
      return false;

  std::string file = folder_ + (accociations_.empty() ? depth_stamps_and_filenames_[i].second : accociations_[i].name1);
  std::string str_png(".png");
  file.replace(file.find(str_png),str_png.length(),".png");
  
  cv::Mat d_img = cv::imread(file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  if(d_img.empty())
      return false;
   
  if (d_img.elemSize() != sizeof(unsigned short))
  {
    std::cout << "Image was not opend in 16-bit format. Please use OpenCV 2.3.1 or higher" << endl;
    exit(1);
  }

  // Datasets are with factor 5000 (pixel to m) 
  // http://cvpr.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps
    
  d_img.convertTo(impl_->depth_buffer, d_img.type(), 0.2);
  depth.data = impl_->depth_buffer.ptr<ushort>();
  depth.cols = impl_->depth_buffer.cols;
  depth.rows = impl_->depth_buffer.rows;
  depth.step = impl_->depth_buffer.cols*depth.elemSize(); // 1280 = 640*2

  if (visualization_)
  {			
    cv::Mat scaled = impl_->depth_buffer/5000.0*65535;	
    cv::imshow("Depth channel", scaled);
    cv::waitKey(3);
  }
  return true;
}

//bool Evaluation::grab (double stamp, PtrStepSz<const unsigned short>& depth, PtrStepSz<const RGB>& rgb24)
bool Evaluation::grab (int stamp, RGBID_SLAM::ImageWrapper<unsigned short>& depth, RGBID_SLAM::ImageWrapper<RGB>& rgb24)
{
  if (accociations_.empty())
  {
    std::cout << "Please set match file" << endl;
    exit(0);
  }
  
  size_t i = static_cast<size_t>(stamp); // temporary solution, now it expects only index

  if ( i>= accociations_.size())
      return false;

  std::string depth_file = folder_ + accociations_[i].name1;
  std::string color_file = folder_ + accociations_[i].name2;
  //std::string str_png(".png");
  //depth_file.replace(depth_file.find(str_png),str_png.length(),".png");
  //color_file.replace(color_file.find(str_png),str_png.length(),".png");

  cv::Mat d_img = cv::imread(depth_file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
  if(d_img.empty())
      return false;
   
  if (d_img.elemSize() != sizeof(unsigned short))
  {
    std::cout << "Image was not opend in 16-bit format. Please use OpenCV 2.3.1 or higher" << endl;
    exit(1);
  }

  // Datasets are with factor 5000 (pixel to m) 
  // http://cvpr.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps
     
  d_img.convertTo(impl_->depth_buffer, d_img.type(), 0.2);
  depth.data = impl_->depth_buffer.ptr<ushort>();
  depth.cols = impl_->depth_buffer.cols;
  depth.rows = impl_->depth_buffer.rows;
  depth.step = impl_->depth_buffer.cols*depth.elemSize(); // 1280 = 640*2

  cv::Mat bgr = cv::imread(color_file);
  if(bgr.empty())
      return false;     
      
  cv::cvtColor(bgr, impl_->rgb_buffer, CV_BGR2RGB);
  
  rgb24.data = impl_->rgb_buffer.ptr<RGB>();
  rgb24.cols = impl_->rgb_buffer.cols;
  rgb24.rows = impl_->rgb_buffer.rows;
  rgb24.step = impl_-> rgb_buffer.cols*rgb24.elemSize();

  return true;  
}

void Evaluation::saveTimeLogFiles(const RGBID_SLAM::VisodoTracker& visodo, const RGBID_SLAM::KeyframeManager& kf_manager, const std::string& kftimes_logfile) const
{
  std::ofstream kftimes_file_stream(kftimes_logfile.c_str());
  kftimes_file_stream.setf(ios::fixed,ios::floatfield);
  
  kftimes_file_stream << "ObtainKeyframe " << "ProcessKeyframeTotal " << "Segmentation " << "DescriptionBoW " << "LoopDetection " << "PoseGraphOptim" << std::endl; 
  
  std::cout << kftimes_logfile << std::endl;
  assert(visodo.kf_times_.size() == kf_manager.segmentation_times_.size());
  std::cout << "saving kf times2" << std::endl;
  
  for (int i=0; i<visodo.kf_times_.size(); i++)
  {
    kftimes_file_stream << visodo.kf_times_[i] << " "
                        << kf_manager.total_times_[i] << " " 
                        << kf_manager.segmentation_times_[i] << " " 
                        << kf_manager.description_times_[i] << " " 
                        << kf_manager.loop_detection_times_[i] << " " 
                        << kf_manager.pose_graph_times_[i] << std::endl;
                        
  }
  
  kftimes_file_stream.close();
  
  
}

void Evaluation::saveAllPoses(const RGBID_SLAM::VisodoTracker& visodo, int frame_number, const std::string& poses_logfile, const std::string& chi_tests_logfile) const
{   
  size_t total = accociations_.empty() ? depth_stamps_and_filenames_.size() : accociations_.size();

  if (frame_number < 0)
      frame_number = (int)total;

  frame_number = std::min(frame_number, (int)visodo.getNumberOfPoses());

  std::cout << "Writing " << frame_number << " poses to " << poses_logfile << endl;
  
  std::ofstream poses_path_file_stream(poses_logfile.c_str());
  poses_path_file_stream.setf(ios::fixed,ios::floatfield);
  
  std::ofstream chi_tests_path_file_stream(chi_tests_logfile.c_str());
  chi_tests_path_file_stream.setf(ios::fixed,ios::floatfield);
  
  float mean_vis_odo_time = 0.f;
  float std_vis_odo_time = 0.f;
  float max_vis_odo_time = 0.f;
  
  for(int i = 0; i < frame_number; ++i)
  {
    mean_vis_odo_time += visodo.getVisOdoTime(i)/frame_number;
    
    if  (visodo.getVisOdoTime(i) > max_vis_odo_time)
      max_vis_odo_time = visodo.getVisOdoTime(i);
  }
  
  for(int i = 0; i < frame_number; ++i)
    std_vis_odo_time += (visodo.getVisOdoTime(i) - mean_vis_odo_time)*(visodo.getVisOdoTime(i) - mean_vis_odo_time) / frame_number;
   
  std_vis_odo_time = sqrt(std_vis_odo_time);
  
  chi_tests_path_file_stream << "Mean time per frame: " << mean_vis_odo_time << std::endl
                         << "Std time per frame: " << std_vis_odo_time << std::endl
                         << "Max time per frame: " << max_vis_odo_time << std::endl;
                         
  std::cout << "Mean time per frame: " << mean_vis_odo_time << std::endl
            << "Std time per frame: " << std_vis_odo_time << std::endl
            << "Max time per frame: " << max_vis_odo_time << std::endl;

  for(int i = 0; i < frame_number; ++i)
  {
    Eigen::Affine3f pose = visodo.getCameraPose(i);
    Eigen::Quaternionf q(pose.rotation());
    Eigen::Vector3f t = pose.translation();
    
    
    float vis_odo_time = visodo.getVisOdoTime(i);

    double stamp = accociations_.empty() ? depth_stamps_and_filenames_[i].first : accociations_[i].time1;

    poses_path_file_stream << stamp << " ";
    poses_path_file_stream << t[0] << " " << t[1] << " " << t[2] << " ";
    poses_path_file_stream << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    
    chi_tests_path_file_stream << stamp << " " << " " << vis_odo_time << endl;
  }
}


void Evaluation::saveAllPoses(std::vector<RGBID_SLAM::Pose> &poses, const RGBID_SLAM::VisodoTracker& visodo, int frame_number, const std::string& poses_logfile, const std::string& chi_tests_logfile) const
{   
  size_t total = accociations_.empty() ? depth_stamps_and_filenames_.size() : accociations_.size();

  if (frame_number < 0)
      frame_number = (int)total;

  frame_number = std::min(frame_number, (int)visodo.getNumberOfPoses());

  std::cout << "Writing " << frame_number << " poses to " << poses_logfile << endl;
  
  std::ofstream poses_path_file_stream(poses_logfile.c_str());
  poses_path_file_stream.setf(ios::fixed,ios::floatfield);
  
  std::ofstream chi_tests_path_file_stream(chi_tests_logfile.c_str());
  chi_tests_path_file_stream.setf(ios::fixed,ios::floatfield);
  
  float mean_vis_odo_time = 0.f;
  float std_vis_odo_time = 0.f;
  float max_vis_odo_time = 0.f;
  
  for(int i = 0; i < frame_number; ++i)
  {
    mean_vis_odo_time += visodo.getVisOdoTime(i)/frame_number;
    
    if  (visodo.getVisOdoTime(i) > max_vis_odo_time)
      max_vis_odo_time = visodo.getVisOdoTime(i);
  }
  
  for(int i = 0; i < frame_number; ++i)
    std_vis_odo_time += (visodo.getVisOdoTime(i) - mean_vis_odo_time)*(visodo.getVisOdoTime(i) - mean_vis_odo_time) / frame_number;
   
  std_vis_odo_time = sqrt(std_vis_odo_time);
  
  chi_tests_path_file_stream << "Mean time per frame: " << mean_vis_odo_time << std::endl
                         << "Std time per frame: " << std_vis_odo_time << std::endl
                         << "Max time per frame: " << max_vis_odo_time << std::endl;
                         
  std::cout << "Mean time per frame: " << mean_vis_odo_time << std::endl
            << "Std time per frame: " << std_vis_odo_time << std::endl
            << "Max time per frame: " << max_vis_odo_time << std::endl;

  for(int i = 0; i < poses.size(); ++i)
  {
    //Eigen::Affine3f pose = visodo.getCameraPose(i);
    Eigen::Quaternionf q(poses[i].rotation_.cast<float>());
    Eigen::Vector3f t = poses[i].translation_.cast<float>();
    
    
    float vis_odo_time = visodo.getVisOdoTime(i);

    double stamp = accociations_.empty() ? depth_stamps_and_filenames_[i].first : accociations_[i].time1;

    poses_path_file_stream << stamp << " ";
    poses_path_file_stream << t[0] << " " << t[1] << " " << t[2] << " ";
    poses_path_file_stream << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    
    chi_tests_path_file_stream << stamp << " " << " " << vis_odo_time << endl;
  }
}


#endif /* HAVE_OPENCV */
