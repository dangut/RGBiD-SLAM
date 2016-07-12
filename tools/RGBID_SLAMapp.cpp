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
#include <signal.h>
//

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/callback_queue.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>

#include <boost/filesystem.hpp>

#include "../include/settings.h"
#include "../include/visodo.h"
#include "../include/keyframe_manager.h"
#include "../include/visualization_manager.h"
#include "../include/util_funcs.h"
#include "../ThirdParty/pcl_gpu_containers/include/initialization.h"

#include <pcl/common/time.h>
//#include <pcl/io/openni_grabber.h>
//#include <pcl/common/angles.h>

#include "../src/internal.h"
#include "evaluation.h"

//using namespace std;
//using namespace pcl;
//using namespace Eigen;

using namespace RGBID_SLAM;


namespace pc = pcl::console;

int RGBID_SLAM::device::dev_id;
cudaDeviceProp RGBID_SLAM::device::dev_prop;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 
struct RGBID_SLAMApp
{
  RGBID_SLAMApp(std::string poses_log = "visodo_poses.txt", std::string misc_log = "visodo_misc.txt") : exit_ (false),   time_ms_ (0)
  {      
  	poses_log_.assign(poses_log);
  	misc_log_.assign(misc_log);     

    //visodo_ = new VisodoTracker(motion_model, sigma_estimator, keyframe_count, finest_level);
    //TODO: input args for each module
    visodo_.reset(new VisodoTracker); 
    std::cout << "visodo ini" << std::endl;
    keyframe_manager_.reset(new KeyframeManager);
    std::cout << "kf manager ini" << std::endl;
    visualization_manager_.reset(new VisualizationManager);
    
    whole_point_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    whole_point_cloud_filtered_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    setPointersBetweenModules();

    frame_counter_ = 0;    
  }

  ~RGBID_SLAMApp()
  {    
  }
  
  void
  loadSettings(std::string const &config_file)
  {
    std::ifstream filestream(config_file.c_str());
    
    if (!filestream.is_open())
    {
      std::cout << "Could not open configuration file " << config_file << std::endl;
      return;
    }
      
    Settings settings(filestream);
    
    visodo_->loadSettings(settings);
    keyframe_manager_->loadSettings(settings);
    //visualization_manager_->loadSettings(settings);    
  }
  
  void
  setPointersBetweenModules()
  {
    visodo_->keyframe_manager_ptr_ = keyframe_manager_;
    //visodo_->visualization_manager_ptr_ = visualization_manager_;
    
    //keyframe_manager_->visodo_ptr_ = visodo_;
    //keyframe_manager_->visualization_manager_ptr_ = visualization_manager_;
    
    visualization_manager_->visodo_ptr_ = visodo_;
    visualization_manager_->keyframe_manager_ptr_ = keyframe_manager_;    
  }
  
  void
  toggleEvaluationMode(const std::string& eval_folder, const std::string& match_file = std::string())
  {
    evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder, match_file) );
    if (!match_file.empty())
      evaluation_ptr_->setMatchFile(match_file);

    visodo_->setRGBIntrinsics (evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);    

    cout << "toggle eval mode" << endl;
  }
  

  void frameRGBDCallback(const sensor_msgs::ImageConstPtr& image_wrapper, const sensor_msgs::ImageConstPtr& depth_wrapper)
  {      
    boost::mutex::scoped_try_lock lock(visodo_->mutex_);
    
    if (!lock)
    {
      return;
    }
    
    (visodo_->depth_).upload(&(depth_wrapper->data[0]), depth_wrapper->step, depth_wrapper->height, depth_wrapper->width);   
    visodo_->timestamp_depth_curr_ = depth_wrapper->header.stamp.toNSec();

   
    (visodo_->rgb24_).upload(&(image_wrapper->data[0]), image_wrapper->step, image_wrapper->height, image_wrapper->width);  
    visodo_->timestamp_rgb_curr_ = image_wrapper->header.stamp.toNSec(); 
    
    visodo_->new_frame_cond_.notify_one();    
  }
  
  void simulateLoopCallback()
  {
    int currentIndex = 0;
    int num_failures = 0;
    visodo_->compute_deltat_flag_ = false;
    while (true)
    {  
      bool grab_success = false;
      
      {   
        boost::mutex::scoped_try_lock lock(visodo_->mutex_);
        
        if (!lock)
          continue;
          
        //std::cout << "locked visodo, grabbing..." << std::endl;        
        
        try { grab_success = evaluation_ptr_->grab(currentIndex, depth_, rgb24_);}
        catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc grabbing" << endl; break; }
        catch (const std::exception& /*e*/) { cout << "Exception grabbing" << endl; break; }   
          
        if (grab_success)
        {
          num_failures = 0;
          (visodo_->depth_).upload (depth_.data, depth_.step, depth_.rows, depth_.cols);     
          (visodo_->rgb24_).upload (rgb24_.data, rgb24_.step, rgb24_.rows, rgb24_.cols);
          (visodo_->new_frame_cond_).notify_one();  
                  
        }
        else
        {
          num_failures+=1;
        }
        
        currentIndex += 1; 
        
        if (num_failures == 10) 
          break;  
      }
      
      if (grab_success)
      {
        //std::cout << "                                          Notifying visodo: " << currentIndex << std::endl;
        
        boost::this_thread::sleep (boost::posix_time::millisec (30)); 
      }
    }
    
    exit_ = true;  
  }
  
  void
  startMainLoopImageSequence ()
  {   
    boost::thread(&RGBID_SLAMApp::simulateLoopCallback,this);
    
    while (!exit_)
    {  
      visualization_manager_->refresh();   
      //boost::this_thread::sleep (boost::posix_time::millisec (10));                       
    } 
    
  }

  void
  startMainLoop (int argc, char* argv[])
  {       
    //ROS subscribing
    ros::init(argc, argv, "visual_odometry_node");
    ros::NodeHandle nh;
    message_filters::Subscriber<sensor_msgs::Image> imageRGB_sub (nh, "/camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub (nh, "/camera/depth/image_raw", 1);     
        
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), imageRGB_sub, depth_sub);
    visodo_->compute_deltat_flag_ = true;
    visodo_->real_time_flag_ = true;
    sync.registerCallback(boost::bind(&RGBID_SLAMApp::frameRGBDCallback, this, _1, _2)); 
    
    //Callback in separate thread. 
    //AsyncSPinner thread grabs frames from the stream if visodo is not busy and notifies visodo thread when new frames arrived
    capture_.reset(new ros::AsyncSpinner(1));
    //ros::spin();   
    
    capture_->start();      
      
    while (nh.ok())
    {  
      visualization_manager_->refresh();                               
    } 
          
    capture_->stop();
      
    //Wait some time to allow child threads to stop
    boost::this_thread::sleep (boost::posix_time::millisec (100)); 
  }   


  bool exit_;  
  int frame_counter_;
  
  boost::shared_ptr<ros::AsyncSpinner> capture_;
  
  VisodoTrackerPtr visodo_;
  KeyframeManagerPtr keyframe_manager_;
  VisualizationManagerPtr visualization_manager_;
  
  Evaluation::Ptr evaluation_ptr_;
  std::string poses_log_;
  std::string misc_log_;
  
  ImageWrapper<PixelRGB> rgb24_;  
  ImageWrapper<unsigned short> depth_;  
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr whole_point_cloud_;  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr whole_point_cloud_filtered_;
  

  int time_ms_;  
  
  void savePosesInFile(const std::string& poses_logfile, const std::string& misc_logfile) const
  {     
    int frame_number = (int)(visodo_->getNumberOfPoses());

    cout << "Writing " << frame_number << " poses to " << poses_logfile << endl;
    
    ofstream poses_path_file_stream(poses_logfile.c_str());
    poses_path_file_stream.setf(ios::fixed,ios::floatfield);
    
    ofstream misc_path_file_stream(misc_logfile.c_str());
    misc_path_file_stream.setf(ios::fixed,ios::floatfield);
    
    float mean_vis_odo_time = 0.f;
    float std_vis_odo_time = 0.f;
    float max_vis_odo_time = 0.f;
    
    for(int i = 0; i < frame_number; ++i)
    {
      mean_vis_odo_time += visodo_->getVisOdoTime(i)/frame_number;
      
      if  (visodo_->getVisOdoTime(i) > max_vis_odo_time)
        max_vis_odo_time = visodo_->getVisOdoTime(i);
    }
    
    for(int i = 0; i < frame_number; ++i)
      std_vis_odo_time = (visodo_->getVisOdoTime(i) - mean_vis_odo_time)*(visodo_->getVisOdoTime(i) - mean_vis_odo_time) / frame_number;
     
    std_vis_odo_time = sqrt(std_vis_odo_time);
    
    misc_path_file_stream << "Mean time per frame: " << mean_vis_odo_time << std::endl
                           << "Std time per frame: " << std_vis_odo_time << std::endl
                           << "Max time per frame: " << max_vis_odo_time << std::endl;
                           
    std::cout << "Mean time per frame: " << mean_vis_odo_time << std::endl
              << "Std time per frame: " << std_vis_odo_time << std::endl
              << "Max time per frame: " << max_vis_odo_time << std::endl;

    for(int i = 0; i < frame_number; ++i)
    {
      Eigen::Affine3f pose = visodo_->getCameraPose(i);
      Eigen::Quaternionf q(pose.rotation());
      Eigen::Vector3f t = pose.translation();
      
      //float chi_test = visodo_->getChiTest(i);
      float vis_odo_time = visodo_->getVisOdoTime(i);

      double stamp = (double) visodo_->getTimestamp(i);

      poses_path_file_stream << stamp << " ";
      poses_path_file_stream << t[0] << " " << t[1] << " " << t[2] << " ";
      poses_path_file_stream << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
      
      misc_path_file_stream << stamp << " " << " " << vis_odo_time << endl;
    }
  } 
  
  void savePointCloudInFile(const std::string pointcloud_file)
  {
    visualization_manager_->getPointCloud(whole_point_cloud_);
    
    std::cout << "Final pc size is: " << whole_point_cloud_->size() << std::endl;
    
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(whole_point_cloud_);
    voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
    voxel_filter.filter(*whole_point_cloud_filtered_);
    
    pcl::PLYWriter writer;
    writer.write(pointcloud_file, *whole_point_cloud_filtered_);
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
print_cli_help ()
{
  cout << "\nVisodo RGBD parameters:" << endl;
  cout << "    --help, -h                          : print this message" << endl;  
  cout << "    -config  <filename>  : Load configuration file  " << endl;
  cout << "    -eval <dataset_folder>  : Evaluation mode for dataset in TUM format " << endl;
  cout << "    -match_file <filename> : Provide file with matches between RGB and depth frames  " << endl;
  cout << "    -gpu <id> : Specify gpu id (in case there is more than one, id=0 by default)  " << endl;
  cout << endl << "";

  return 0;
}


int
main (int argc, char* argv[])
{    
  if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
    return print_cli_help ();
    
  //Settings settings_SLAM;
  //TODO: pc::parse for settings file
  RGBID_SLAM::device::dev_id = 0;
  pc::parse_argument (argc, argv, "-gpu", RGBID_SLAM::device::dev_id);
  pcl::gpu::setDevice (RGBID_SLAM::device::dev_id);
  
  cudaSafeCall( cudaGetDeviceProperties(&RGBID_SLAM::device::dev_prop, RGBID_SLAM::device::dev_id) );
  pcl::gpu::printShortCudaDeviceInfo (RGBID_SLAM::device::dev_id);
  
  std::string poses_logfile("poses");
  std::string misc_logfile("misc");
  std::string kf_times_logfile("kf_times");
  std::string pointcloud_file("pointclod");
  
  std::string config_file="";
  pc::parse_argument (argc, argv, "-config", config_file);
  
  std::string calib_file="";
  pc::parse_argument (argc, argv, "-calib", calib_file);
  
  std::cout << "Memory usage BEFORE starting" << std::endl;
	RGBID_SLAM::device::showGPUMemoryUsage();

	std::cout << "init pointer grabber visodo" << std::endl;
  
  //Check if we are running in evaluation mode (offline sequences of images)
  std::string eval_folder, match_file;
  
  if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
  {
      //init data source latter
      pc::parse_argument (argc, argv, "-match_file", match_file);
      std::size_t found_last = eval_folder.find_last_of("/\\");
      std::string eval_folder2 = eval_folder.substr(0,found_last);
      
      std::size_t found_prelast = eval_folder2.find_last_of("/\\");
      std::string dataset_name = eval_folder2.substr(found_prelast+1);
      
      poses_logfile = dataset_name + "_" + poses_logfile;  
  	  misc_logfile = dataset_name + "_"  + misc_logfile;  
      kf_times_logfile = dataset_name + "_"  + kf_times_logfile;
      pointcloud_file = dataset_name + "_"  + pointcloud_file;
  }   
  /////////////////////////////////////////////////////////////////////
  
  poses_logfile.append(".txt");
  misc_logfile.append(".txt");
  kf_times_logfile.append(".txt");
  pointcloud_file.append(".ply");
  	
  std::cout << "init visodoApp" << std::endl;
  
  {
    RGBID_SLAMApp app (poses_logfile, misc_logfile);
    
    if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
      app.toggleEvaluationMode(eval_folder, match_file);  
    
    if (!config_file.empty())
    {
      app.loadSettings(config_file);
    }
    
    if (!calib_file.empty())
    {
      (app.visodo_)->loadCalibration(calib_file);
    }
    
    std::cout << "end init visodoApp" << std::endl;
    printf("\aBeep!\n" );
      
    std::cout << "Loading vocabulary..." << std::endl;
    (app.keyframe_manager_)->loop_closer_ptr_->loadVocabulary();
    
    
    (app.keyframe_manager_)->start();
    (app.visodo_)->start();
    //Visualization will run in the main thread. Dont start
    
    boost::this_thread::sleep (boost::posix_time::millisec (1)); 
    
    
    if (app.evaluation_ptr_)
    {
       std::cout << "starting main loop image sequence\n";
       try { app.startMainLoopImageSequence(); }      
       catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
       catch (const std::exception& /*e*/) { cout << "Exception im seq" << endl; } 
       app.evaluation_ptr_->saveAllPoses(*(app.visodo_), -1, poses_logfile, misc_logfile);      
    }
    else
    {  
      try { app.startMainLoop (argc, argv); }      
      catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
      catch (const std::exception& /*e*/) { cout << "Exception" << endl; }
      app.savePosesInFile(poses_logfile,  misc_logfile);
    }
    
    
    //(app.visodo_)->stop();
    
    (app.keyframe_manager_)->stop();
    (app.keyframe_manager_)->join();
    
    //if ((app.keyframe_manager_)->loop_found_ == true)
    //{
      //(app.keyframe_manager_)->performOptimisation();
      //(app.keyframe_manager_)->loop_found_ = false;
    //}
    
    std::string poses_logfile2 = poses_logfile;
    std::string str_txt(".txt");
    poses_logfile2.replace(poses_logfile2.find(str_txt),str_txt.length(),"_loopsClosed.txt");
    
    if (app.evaluation_ptr_)
    {
      app.evaluation_ptr_->saveAllPoses((app.keyframe_manager_)->poses_, *(app.visodo_),-1, poses_logfile2, misc_logfile); 
      app.evaluation_ptr_->saveTimeLogFiles(*(app.visodo_), *(app.keyframe_manager_) , kf_times_logfile);  
    }
    
    (app.visodo_).reset();
    
    //(app.keyframe_manager_)->keyframes_list_.clear();
    (app.keyframe_manager_).reset();     
    
    //std::cout << "press key to release vis_manager ptr" << std::endl;
    //std::cin.ignore();
    printf("\aBeep!\n" );
    
    //while (true)
    {  
      (app.visualization_manager_)->refresh();   
      //boost::this_thread::sleep (boost::posix_time::millisec (10));                       
    } 
    
    app.savePointCloudInFile(pointcloud_file);
    
    (app.visualization_manager_).reset();
    //std::cin.ignore();
  }
  
  
  std::cout << "visodo exiting...\n";
  return 0;
}
