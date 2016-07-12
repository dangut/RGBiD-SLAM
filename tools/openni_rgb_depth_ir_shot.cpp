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
 * This is a simple program to get RGB, depth and IR images mainly for
 * calibration of RGB-D cameras. Upon pressing 's' a shot of each of the 
 * 3 streams is saved. 
 * 
 * It is not possible to obtain synchronised RGB and IR streams, and thus the
 * shots of each stream are taken consecutively. As a consequence to obtain a 
 * good stereo calibration one should design an experimental setup where the 
 * pattern and the camera can be rigidly fixed before taking one capture.
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/common/time.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <time.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/png_io.h>


#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
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
static const double A_FACTOR = -65535.0/10000.0;
static const double B_FACTOR = 65535.0;

namespace pclGrabber
{
	struct PixelRGB
	{ 
		unsigned char r, g, b;
	};
}

class OpenNIShoter
{
  public:
    bool save;
    std::string write_folder_;
    
    cv::Mat cv_depth_, cv_rgb_, cv_ir_, cv_ir_raw_;
    std::ofstream off_depth_; 
    std::ofstream off_rgb_; 
    std::ofstream off_ir_; 
    std::ofstream off_stamps_;
    boost::shared_ptr<pcl::visualization::ImageViewer> image_viewer_;
    boost::shared_ptr<pcl::visualization::ImageViewer> depth_viewer_;
    boost::shared_ptr<pcl::visualization::ImageViewer> ir_viewer_;
    int cols_;
    int rows_;
    long int timestamp_rgb_old;
    //boost::mutex image_mutex_, viewer_mutex_;
    char depth_file[128], depth_file_PNG[128];
    char rgb_file[128], rgb_file_PNG[128];
    char ir_file[128], ir_file_PNG[128];
    bool exit_;
    bool get_image_;
    long int timestamp_rgb_, timestamp_depth_, timestamp_ir_;    
    
    std::vector<pclGrabber::PixelRGB> rgb_data_;
    std::vector<unsigned short> depth_data_;
    std::vector<unsigned short> ir_data_;
    
    boost::mutex rgb_mutex_;
    boost::mutex depth_mutex_;
    boost::mutex ir_mutex_;
    
    bool new_rgb_, new_depth_, new_ir_;
      
    boost::shared_ptr<ros::AsyncSpinner> capture_;


    OpenNIShoter ( std::string write_folder) : write_folder_(write_folder) {}

    void
    openLogFiles()
    {
	    std::string depth_txt = write_folder_ + "/depth.txt";
      std::string rgb_txt = write_folder_ + "/rgb.txt";
      std::string ir_txt = write_folder_ + "/ir.txt";

      off_depth_.open(depth_txt.c_str());
      off_rgb_.open(rgb_txt.c_str());
      off_ir_.open(ir_txt.c_str());

      if (!off_depth_.is_open() || !off_rgb_.is_open()|| !off_ir_.is_open())
      {
        std::cout << "Can't open files" << std::endl;
        return;
      }

      off_depth_ << "# depth maps" << std::endl
                  << "# sequence folder: ' " << write_folder_ << " '" << std::endl
                  << "# timestamp({mu}s)    filename" << std::endl;

      off_rgb_ << "# RGB maps" << std::endl
                << "# sequence folder: ' " << write_folder_ << " '" << std::endl
                << "# timestamp({mu}s)    filename" << std::endl;
                
      off_ir_ << "# IR maps" << std::endl
                << "# sequence folder: ' " << write_folder_ << " '" << std::endl
                << "# timestamp({mu}s)    filename" << std::endl;

    }
	
    static void 
    keyboardCallback(const pcl::visualization::KeyboardEvent &e, void *cookie)
    {
      OpenNIShoter* recorder = reinterpret_cast<OpenNIShoter*> (cookie);
      
      int key = e.getKeyCode ();
      
      if (e.keyUp())
      {
        if ((key == 'q') || (key == 'Q'))
        {
          recorder->exit_ = true;
        }
        else if ((key == 's') || (key == 'S'))
        {
          recorder->get_image_ = true;
        }
      }
    }

    void 
    frameRGBCallback (const sensor_msgs::ImageConstPtr& rgb_wrapper)
    {
      //
      static unsigned count = 0;
	    count++;
      
      if (count == 1)
      {
        std::cout << "RGB height:" << rgb_wrapper->height << std::endl;
        std::cout << "RGB width:" << rgb_wrapper->width << std::endl;
        std::cout << "RGB encoding:" << rgb_wrapper->encoding << std::endl;
        std::cout << "RGB step:" << rgb_wrapper->step << std::endl;
        std::cout << std::endl;
      }	  
      
      {  
        boost::mutex::scoped_lock lock(rgb_mutex_);   
	      int cols = rgb_wrapper->width;
	      int rows =  rgb_wrapper->height;          
        
        memcpy((unsigned char*)&rgb_data_[0],(unsigned char*)&(rgb_wrapper->data[0]),3*cols*rows*sizeof(unsigned char));
        
        new_rgb_ = true;
        timestamp_rgb_ = (long int) (rgb_wrapper->header.stamp.toNSec());	  
	    }	     	
	     
       
    }
    
    
    void 
    frameDepthCallback (const sensor_msgs::ImageConstPtr& depth_wrapper)
    {
      static unsigned count = 0;
	    count++;
      
      if (count == 1)
      {        
        std::cout << "depth height:" << depth_wrapper->height << std::endl;
        std::cout << "depth width:" << depth_wrapper->width << std::endl;
        std::cout << "depth encoding:" << depth_wrapper->encoding << std::endl;
        std::cout << "depth step:" << depth_wrapper->step << std::endl;
        std::cout << std::endl;
      }	  
      
      {    
        boost::mutex::scoped_lock lock(depth_mutex_);   
	      int cols = depth_wrapper->width;
	      int rows =  depth_wrapper->height;          
	      
        memcpy(&depth_data_[0],&(depth_wrapper->data[0]),cols*rows*sizeof(unsigned short));
        
        new_depth_ = true;
        
	      for (int i=0; i < cols*rows; i++)
	      	depth_data_[i] = 5 * depth_data_[i];
          
        timestamp_depth_ = (long int) (depth_wrapper->header.stamp.toNSec());
	    }	  
         	
      
    }
    
    
    void 
    frameIRCallback (const sensor_msgs::ImageConstPtr& ir_wrapper)
    {
      static unsigned count = 0;
	    count++;
      
      if (count == 1)
      {        
        std::cout << "IR height:" << ir_wrapper->height << std::endl;
        std::cout << "IR width:" << ir_wrapper->width << std::endl;
        std::cout << "IR encoding:" << ir_wrapper->encoding << std::endl;
        std::cout << "IR step:" << ir_wrapper->step << std::endl;
        std::cout << std::endl;
      }	  
      
      {    
        boost::mutex::scoped_lock lock(ir_mutex_);   
	      int cols = ir_wrapper->width;
	      int rows =  ir_wrapper->height;          
	      
        memcpy(&ir_data_[0],&(ir_wrapper->data[0]),cols*rows*sizeof(unsigned short));	 
        new_ir_ = true;    
        
        //ir image values have 10bits ambedded in a format of 16bits
        //for (int i=0; i < cols*rows; i++)
	      	//ir_data_[i] =  (ir_data_[i] + 1)/4 -1;
           
        timestamp_ir_ = (long int) (ir_wrapper->header.stamp.toNSec());
	    }	  
      
    }

    
    
    
    void saveImagesToDisk(ros::NodeHandle& nh, ros::Subscriber& sub)
    {	 
    	//pcl::ScopeTime t1 ("save images");
    	  {  
          boost::mutex::scoped_lock lock(rgb_mutex_);   
          memcpy( cv_rgb_.data, &rgb_data_[0], 3*rows_*cols_*sizeof(unsigned char));
          
          new_rgb_ = false;
        }
        
        capture_->stop();        
        sub.shutdown();        
        sub = nh.subscribe("/camera/depth/image_raw", 1, &OpenNIShoter::frameDepthCallback, this);        
        new_depth_ = false;
        capture_->start();
        
        do
        {         
          if (new_depth_ == true )
          {
            boost::mutex::scoped_lock lock(depth_mutex_);   
            memcpy( cv_depth_.data, &depth_data_[0], rows_*cols_*sizeof(uint16_t));
          
            new_depth_ = false;
            break;
          }
          
          boost::this_thread::sleep (boost::posix_time::millisec (10)); 
             
        } while (1);
        
        
        capture_->stop();        
        sub.shutdown();        
        sub = nh.subscribe("/camera/ir/image", 1, &OpenNIShoter::frameIRCallback, this);   
        new_ir_ = false;         
        capture_->start();
        
        do
        {         
          if (new_ir_ == true )
          {            
            boost::mutex::scoped_lock lock(ir_mutex_);   
            memcpy( cv_ir_raw_.data, &ir_data_[0], rows_*cols_*sizeof(uint16_t));            
            
            cv_ir_raw_.convertTo(cv_ir_, CV_8U); 
            cv::equalizeHist( cv_ir_, cv_ir_ );
            
            int max = 0;
            
            for (int i=0; i< rows_; i++)
            {
              for (int j=0; j< cols_; j++)
              {
               if (ir_data_[i+j*rows_] > max)
                max = ir_data_[i+j*rows_];
              }
            }
            
            std::cout << "max IR val: " << max << std::endl;
            
          
            new_ir_ = false;
            break;
          }
          
          boost::this_thread::sleep (boost::posix_time::millisec (10)); 
             
        } while (1);
        
        capture_->stop();        
        sub.shutdown();        
        sub = nh.subscribe("/camera/rgb/image_raw", 1, &OpenNIShoter::frameRGBCallback, this);         
        capture_->start();

	      sprintf(depth_file, "/Depth/%018ld.png", timestamp_depth_ );
	      sprintf(rgb_file, "/RGB/%018ld.png", timestamp_rgb_);  
        sprintf(ir_file, "/IR/%018ld.png", timestamp_ir_);  
        
	      cv::imwrite( write_folder_ + depth_file, cv_depth_);
	      cv::imwrite( write_folder_ + rgb_file, cv_rgb_);
        cv::imwrite( write_folder_ + ir_file, cv_ir_);
	      
	      sprintf(depth_file_PNG, "/Depth/%018ld.png", timestamp_depth_);
	      sprintf(rgb_file_PNG, "/RGB/%018ld.png", timestamp_rgb_);
        sprintf(ir_file_PNG, "/IR/%018ld.png", timestamp_ir_);

	      off_rgb_.width(18);
	      off_rgb_.fill('0');
	      off_rgb_ << timestamp_rgb_;
	      off_rgb_ <<  " " << rgb_file_PNG << std::endl;
        
        off_depth_.width(18);
	      off_depth_.fill('0');
	      off_depth_ << timestamp_depth_;
	      off_depth_ <<  " " << depth_file_PNG << std::endl;
        
        off_ir_.width(18);
	      off_ir_.fill('0');
	      off_ir_ << timestamp_ir_;
	      off_ir_ <<  " " << ir_file_PNG << std::endl;
	  }
    

    void 
    run (int argc, char* argv[])
    {
      exit_ = false;
      get_image_ = false;
      cols_ = 640;
      rows_ = 480;
      new_rgb_ = false;
      new_depth_ = false;
      new_ir_ = false;
      
      cv_rgb_.create(480, 640, CV_8UC3);
      cv_depth_.create(480, 640, CV_16UC1);
      cv_ir_.create(480, 640, CV_16UC1);
      cv_ir_raw_.create(480, 640, CV_16UC1);
      
      rgb_data_.resize(480 *640);
      depth_data_.resize(480 *640);      
      ir_data_.resize(480 *640);
	      
      image_viewer_.reset (new pcl::visualization::ImageViewer ("PCL OpenNI image"));
      image_viewer_->setPosition (0, 0);
      image_viewer_->setSize (640, 480);
      image_viewer_->registerKeyboardCallback(keyboardCallback,(void*)this);
         
       openLogFiles();
       
      //ROS subscribing
      ros::init(argc, argv, "capture_node");
      ros::NodeHandle nh;
      ros::Subscriber sub = nh.subscribe("/camera/rgb/image_raw", 1, &OpenNIShoter::frameRGBCallback, this);         
      
      //Callback in separate thread. 
      //AsyncSPinner thread grabs frames from the stream if visodo is not busy and notifies visodo thread when new frames arrived
      capture_.reset(new ros::AsyncSpinner(1));
      //ros::spin();   
      
      capture_->start(); 
     
      std::cout << "<Esc>, \'q\', \'Q\': quit the program" << std::endl;
      std::cout << "\' \': pause" << std::endl;
      std::cout << "\'s\': get image (if in snapshot mode)" << std::endl;
      char key;

      
      do
      {         
        if (new_rgb_ == true )
        {
          //std::cout <<"new rgb" << std::endl;
          if (get_image_) 
          {
            saveImagesToDisk(nh, sub);
            get_image_ = false;
          }
          else
          {
            boost::mutex::scoped_lock lock(rgb_mutex_);   
            
            if (&rgb_data_[0] != 0)
              image_viewer_->addRGBImage ((unsigned char*)&rgb_data_[0], cols_, rows_);
            
            new_rgb_ = false;
          }
        }
        
        //boost::this_thread::sleep (boost::posix_time::millisec (10)); 
        
        image_viewer_->spinOnce ();
           
      } while (!exit_);

      // stop the grabber
      capture_->stop ();
    }
};

int
main (int argc, char **argv)
{
  char write_folder_c[1024];
  
  std::string write_folder_prefix = "sequence";

  time_t now = time(0);
  struct tm tstruct;
  tstruct = *localtime(&now);
  int year = 1900 + tstruct.tm_year;
  int month = tstruct.tm_mon;
  int day = tstruct.tm_mday;
  int hour = tstruct.tm_hour;
  int min = tstruct.tm_min;
  int sec = tstruct.tm_sec;
  
  if (argc > 1)
    write_folder_prefix = argv[1];
  else
    printf("Usage: openni_grabber_save_images  my_write_folder_prefix snapshot_mode_flag{0,1}");

  sprintf(write_folder_c, "%s_%02d_%02d_%04d__%02dh_%02dm_%02ds",write_folder_prefix.c_str(),day,month,year,hour,min,sec);
  
  std::string write_folder(write_folder_c);
  mkdir(write_folder_c,0777);
  mkdir((write_folder + "/Depth").c_str(),0777);
  mkdir((write_folder + "/RGB").c_str(),0777);
  mkdir((write_folder + "/IR").c_str(),0777);

  OpenNIShoter v (write_folder);
    
  v.run (argc,argv);
  return (0);
}
