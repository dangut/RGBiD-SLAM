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

#pragma once

#include <string>
#include <boost/shared_ptr.hpp>
#include <../include/types.h>
#include <../include/keyframe_manager.h>
#include <../include/visodo.h>
#include <../include/visualization_manager.h>


/** \brief  class for  RGB-D SLAM Dataset and Benchmark
  * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
  */
class Evaluation
{
public:
  typedef boost::shared_ptr<Evaluation> Ptr; 
  typedef RGBID_SLAM::PixelRGB RGB;
  //typedef RGBID_SLAM::ImageWrapper ImageWrapper;

  Evaluation(const std::string& folder, const std::string& match_file);

  /** \brief Sets file with matches between depth and rgb */
  void setMatchFile(const std::string& file);

  void associate_depth_rgb(const std::string& file_depth, const std::string& file_rgb);

  /** \brief Reads rgb frame from the folder   
    * \param stamp index of frame to read (stamps are not implemented)
    */
  bool grab (double stamp, RGBID_SLAM::ImageWrapper<RGB>& rgb24);

  /** \brief Reads depth frame from the folder
    * \param stamp index of frame to read (stamps are not implemented)
    */
  bool grab (double stamp, RGBID_SLAM::ImageWrapper<unsigned short>& depth);

  /** \brief Reads depth & rgb frame from the folder. Before calling this folder please call 'setMatchFile', or an error will be returned otherwise.
    * \param stamp index of accociated frame pair (stamps are not implemented)
    */
  //bool grab (double stamp, pcl::gpu::PtrStepSz<const unsigned short>& depth, pcl::gpu::PtrStepSz<const RGB>& rgb24);
  
  bool grab (int stamp, RGBID_SLAM::ImageWrapper<unsigned short>& depth, RGBID_SLAM::ImageWrapper<RGB>& rgb24);
  
  const static float fx, fy, cx, cy;
  
  void saveTimeLogFiles(const RGBID_SLAM::VisodoTracker& visodo, const RGBID_SLAM::KeyframeManager& kf_manager, const std::string& kftimes_logfile) const;

  void saveAllPoses(const RGBID_SLAM::VisodoTracker& visodo, int frame_number = -1, const std::string& poses_logfile = "visodo_poses.txt", const std::string& chi_tests_logfile = "visodo_chi_tests.txt") const;
  
  void saveAllPoses(std::vector<RGBID_SLAM::Pose> &poses, const RGBID_SLAM::VisodoTracker& visodo, int frame_number = -1, const std::string& poses_logfile = "visodo_poses.txt", const std::string& chi_tests_logfile = "visodo_chi_tests.txt") const;

private:
  std::string folder_;
  bool visualization_;

  std::vector< std::pair<double, std::string> > rgb_stamps_and_filenames_;
  std::vector< std::pair<double, std::string> > depth_stamps_and_filenames_;

  struct Association
  {
    double time1, time2;
    std::string name1, name2;
  };

  std::vector< Association > accociations_;

  void readFile(const std::string& file, std::vector< std::pair<double, std::string> >& output);

  struct Impl;
  boost::shared_ptr<Impl> impl_;
};

