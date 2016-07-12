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

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/common/time.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <time.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/png_io.h>


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
static const double A_FACTOR = -65535.0/10000.0;
static const double B_FACTOR = 65535.0;

namespace pclGrabber
{
	struct PixelRGB
	{ 
		unsigned char r, g, b;
	};
	
  
	class queue_rgbd
	{
	  private:
	  	std::deque< std::vector<pclGrabber::PixelRGB> > rgb_q_;
	  	std::deque< std::vector<unsigned short> > depth_q_;
	  	std::deque< long int > timestamps_rgb_;
	  	std::deque< long int > timestamps_depth_;
	  	boost::mutex m_;
	  	int max_size_;
	  	//boost::condition_variable full_;
	  	boost::condition_variable c_empty_;
	  	boost::condition_variable c_full_;
	  
	  public:
	  
	  	void initialise()
	  	{
	  		rgb_q_.clear();
	  		depth_q_.clear();
	  		timestamps_rgb_.clear();
	  		timestamps_depth_.clear();
	  		max_size_ = 500;
	  	}
	  	
	  	void push(const std::vector<pclGrabber::PixelRGB> &rgb_data, const std::vector<unsigned short> &depth_data, long int timestamp_rgb, long int timestamp_depth)
	  	{
	  		//pcl::ScopeTime t1 ("push");
	  		boost::mutex::scoped_lock lock(m_);
	  		
	  		while (rgb_q_.size() == max_size_){
	  			c_full_.wait(lock);
	  			std::cout << "Buffer is full!!" << std::endl;
	  		}
	  			
	  		rgb_q_.push_back(rgb_data);
	  		depth_q_.push_back(depth_data);
	  		timestamps_rgb_.push_back(timestamp_rgb);
	  		timestamps_depth_.push_back(timestamp_depth);
	  		
	  		/*
	  		if (rgb_q_.size() > 1){
		  		std::cout << "Buffer size: " << rgb_q_.size() << std::endl;
		  		std::cout << "At timestamp : " << timestamp_rgb << std::endl;
		  		std::cout << " " << std::endl;
		  	}*/
	  		
	  		c_empty_.notify_one();
	  	}
	  	
	  	void pop( std::vector<pclGrabber::PixelRGB> &rgb_data, std::vector<unsigned short> &depth_data, long int &timestamp_rgb, long int &timestamp_depth)
	  	{
	  		//pcl::ScopeTime t1 ("pop");
	  		boost::mutex::scoped_lock lock(m_);
	  		
	  		while (rgb_q_.size() == 0)
	  			c_empty_.wait(lock);
	  			
	  		rgb_data = rgb_q_.front();
	  		depth_data = depth_q_.front();
	  		timestamp_rgb = timestamps_rgb_.front();
	  		timestamp_depth = timestamps_depth_.front();
	  		
	  		rgb_q_.pop_front();
	  		depth_q_.pop_front();
	  		timestamps_rgb_.pop_front();
	  		timestamps_depth_.pop_front();
	  		
	  		c_full_.notify_one();
	  	}
      
      void clear()
      {
        boost::mutex::scoped_lock lock(m_);
        rgb_q_.clear();
        depth_q_.clear();
        c_full_.notify_one();
      }
	  };
    
}

class SimpleOpenNIRecorder
{
  public:
    bool save;
    std::string write_folder_;
    std::vector<unsigned short> producer_depth_data_, consumer_depth_data_;
    std::vector<pclGrabber::PixelRGB> producer_rgb_data_, consumer_rgb_data_, rgb_data;
    pclGrabber::queue_rgbd buffer_rgbd_;
    cv::Mat cv_depth_, cv_rgb_;
    std::ofstream off_depth_; 
    std::ofstream off_rgb_; 
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
    bool exit_;
    bool pause_;
    int snapshot_mode_;
    bool get_image_;
    long int timestamp_rgb_, timestamp_depth_;
      
    boost::shared_ptr<ros::AsyncSpinner> capture_;


    SimpleOpenNIRecorder ( std::string write_folder) : write_folder_(write_folder) {}

    void
    openLogFiles()
    {
	    std::string depth_txt = write_folder_ + "/depth.txt";
      std::string rgb_txt = write_folder_ + "/rgb.txt";
      std::string stamp_diffs_txt = write_folder_ + "/timestamps_diffs.txt";

      off_depth_.open(depth_txt.c_str());
      off_rgb_.open(rgb_txt.c_str());
      off_stamps_.open(stamp_diffs_txt.c_str());

      if (!off_depth_.is_open() || !off_rgb_.is_open())
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

    }
	
    static void 
    keyboardCallback(const pcl::visualization::KeyboardEvent &e, void *cookie)
    {
      SimpleOpenNIRecorder* recorder = reinterpret_cast<SimpleOpenNIRecorder*> (cookie);
      
      int key = e.getKeyCode ();
      
      if (e.keyUp())
      {
        if (key == ' ')
        {
          if (recorder->pause_ == true)
            recorder->pause_ = false;
          else
            recorder->pause_ = true;
        }
        else if ((key == 'q') || (key == 'Q'))
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
    frameRGBDCallback (const sensor_msgs::ImageConstPtr& rgb_wrapper, const sensor_msgs::ImageConstPtr& depth_wrapper)
    {
      //
      static unsigned count = 0;
      static long timestamp_offset = std::min(depth_wrapper->header.stamp.toNSec(), rgb_wrapper->header.stamp.toNSec());
      //boost::mutex::scoped_lock lock (image_mutex_);
	    count++;
      
      if (count == 1)
      {
        std::cout << "RGB height:" << rgb_wrapper->height << std::endl;
        std::cout << "RGB width:" << rgb_wrapper->width << std::endl;
        std::cout << "RGB encoding:" << rgb_wrapper->encoding << std::endl;
        std::cout << "RGB step:" << rgb_wrapper->step << std::endl;
        std::cout << std::endl;
        std::cout << "depth height:" << depth_wrapper->height << std::endl;
        std::cout << "depth width:" << depth_wrapper->width << std::endl;
        std::cout << "depth encoding:" << depth_wrapper->encoding << std::endl;
        std::cout << "depth step:" << depth_wrapper->step << std::endl;
      }
	  
      if (count != 1)
      {      	
        //pcl::ScopeTime t1 ("grab images");
	      int cols = depth_wrapper->width;
	      int rows =  depth_wrapper->height;          
	      
        memcpy(&producer_depth_data_[0],&(depth_wrapper->data[0]),cols*rows*sizeof(unsigned short));
        memcpy((unsigned char*)&producer_rgb_data_[0],(unsigned char*)&(rgb_wrapper->data[0]),3*cols*rows*sizeof(unsigned char));
	      
	      for (int i=0; i < cols*rows; i++)
	      	producer_depth_data_[i] = 5 * producer_depth_data_[i];
	    }
	     	
	      long int timestamp_depth = (long int) (depth_wrapper->header.stamp.toNSec() - timestamp_offset);
	      long int timestamp_rgb = (long int) (rgb_wrapper->header.stamp.toNSec() - timestamp_offset);	    
	      	
	      buffer_rgbd_.push(producer_rgb_data_, producer_depth_data_, timestamp_rgb, timestamp_depth);  
	      
    }

     //std::cout << " " << std::endl;
    
    
    void saveImagesToDisk()
    {	 
    	//pcl::ScopeTime t1 ("save images");
    	  
	      memcpy( cv_depth_.data, &consumer_depth_data_[0], rows_*cols_*sizeof(uint16_t));
	      memcpy( cv_rgb_.data, &consumer_rgb_data_[0], 3*rows_*cols_*sizeof(unsigned char));
	      
	      //cv::Mat depth_im;
	      //cv_depth_.convertTo(depth_im, CV_16UC1, 5.0);
	      cv::cvtColor(cv_rgb_, cv_rgb_, CV_BGR2RGB);

	      sprintf(depth_file, "/Depth/%018ld.ppm", timestamp_depth_ );
	      sprintf(rgb_file, "/RGB/%018ld.ppm", timestamp_rgb_);      
	      

	      //cv::imwrite( write_folder_ + depth_file, depth_im);
	      cv::imwrite( write_folder_ + depth_file, cv_depth_);
	      cv::imwrite( write_folder_ + rgb_file, cv_rgb_);
	      
	      sprintf(depth_file_PNG, "/Depth/%018ld.ppm", timestamp_depth_);
	      sprintf(rgb_file_PNG, "/RGB/%018ld.ppm", timestamp_rgb_);
	      
	      off_depth_.width(18);
	      off_depth_.fill('0');
	      off_depth_ << timestamp_depth_;
	      off_depth_ <<  " " << depth_file_PNG << std::endl;

	      off_rgb_.width(18);
	      off_rgb_.fill('0');
	      off_rgb_ << timestamp_rgb_;
	      off_rgb_ <<  " " << rgb_file_PNG << std::endl;
	      
	      off_stamps_.width(18);
	      off_stamps_.fill('0');
	      off_stamps_  << timestamp_rgb_ - timestamp_rgb_old << std::endl;
	      
	      timestamp_rgb_old = timestamp_rgb_;
	      
	}

    void 
    run (int argc, char* argv[])
    {
      save = false;
      pause_ = false;
      exit_ = false;
      get_image_ = false;
      cols_ = 640;
      rows_ = 480;
              
      timestamp_rgb_old = 0;
	
      cv_depth_.create(480, 640, CV_16UC1);
      cv_rgb_.create(480, 640, CV_8UC3);
      producer_depth_data_.resize(480 *640);
      producer_rgb_data_.resize(480 *640);
      consumer_depth_data_.resize(480 *640);
      consumer_rgb_data_.resize(480 *640);
	      
      image_viewer_.reset (new pcl::visualization::ImageViewer ("PCL OpenNI image"));
      image_viewer_->setPosition (0, 0);
      image_viewer_->setSize (640, 480);
      image_viewer_->registerKeyboardCallback(keyboardCallback,(void*)this);
      
      depth_viewer_.reset (new pcl::visualization::ImageViewer ("PCL OpenNI depth"));
      depth_viewer_->setPosition (640, 0);
      depth_viewer_->setSize (640, 480);
      depth_viewer_->registerKeyboardCallback(keyboardCallback,(void*)this);
            
      buffer_rgbd_.initialise();
      
      //ROS subscribing
      ros::init(argc, argv, "capture_node");
      ros::NodeHandle nh;
      
      rgb_data.resize(640*480);
      openLogFiles();
      
      message_filters::Subscriber<sensor_msgs::Image> imageRGB_sub (nh, "/camera/rgb/image_raw", 1);
      message_filters::Subscriber<sensor_msgs::Image> depth_sub (nh, "/camera/depth/image_raw", 1);     
          
      typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
      message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), imageRGB_sub, depth_sub);
      sync.registerCallback(boost::bind(&SimpleOpenNIRecorder::frameRGBDCallback, this, _1, _2)); 
      
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
        if (!pause_)
        {
          
          buffer_rgbd_.pop(consumer_rgb_data_, consumer_depth_data_, timestamp_rgb_, timestamp_depth_);
          
          if (!(snapshot_mode_) || (get_image_) )
          {
            pcl::ScopeTime t1 ("record frame");
            saveImagesToDisk();
            get_image_ = false;
          }

          if (&consumer_rgb_data_[0] != 0)
            image_viewer_->addRGBImage ((unsigned char*)&consumer_rgb_data_[0], cols_, rows_);	
            depth_viewer_->addShortImage ((unsigned short*)&consumer_depth_data_[0], cols_, rows_);	
        }
        else
        {
          buffer_rgbd_.clear();
          boost::this_thread::sleep (boost::posix_time::millisec (10)); 
        }
        image_viewer_->spinOnce ();
        depth_viewer_->spinOnce ();   
           
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

  SimpleOpenNIRecorder v (write_folder);
  
  v.snapshot_mode_ = 0;
  
  if (argc > 2)
    v.snapshot_mode_ = atoi(argv[2]);
    
  v.run (argc,argv);
  return (0);
}
