#RGBiD-SLAM

RGBiD-SLAM is a direct and dense visual SLAM algorithm for RGB-D cameras running in CPU and GPU. The motion of the RGB-D camera is tracked by pixelwise minimisation of the photometric error as well as the geometric error expressed in inverse depth (iD) of the current frame with respect to a reference view. A smooth dense 3D reconstruction of the environment is computed by fusing close raw frames into one single keyframe. The system uses a Bag of Words approach to close loops and improve the reconstrction and provides also in real-time a segmentation of the scene based on the structure of its elements.

#Related Publications

[1] D. Gutierrez-Gomez, W. Mayol-Cuevas, J.J. Guerrero. Inverse Depth for Accurate Photometric and Geometric Error Minimisation in RGB-D Dense Visual Odometry. In Proc. IEEE/RSJ Int. Conf. on Robotics and Automation (ICRA). 2015.

[2] D. Gutierrez-Gomez, W. Mayol-Cuevas, J.J. Guerrero. Dense rgb-d visual odometry using inverse depth. Robotics and Autonomous Systems (RAS), 75(Part B):571 – 583. Special Section on 3D Perception with PCL. 2016.

#License
RGBiD-SLAM is licensed under GNU General Public License Version 3 (see LICENSE.txt)

#Dependencies

This code has been compiled and executed succesfully in Ubuntu 12.04 64 bits system with ros hydro and Ubuntu 14.04 64 bits with ros indigo.
Apart from off-line sequences recorded in the TUM dataset format, on-line it has been tested only with an Asus Xtion Pro Live RGB-D camera.
At the moment I cannot guarantee that the system will work with more recent Ubuntu or ROS distros or other RGB-D sensors.
The library requires a CUDA capable NVIDIA GPU. The code has been compiled and works under 5.0 architecture but should work also for 2.x, 3.x and 5.x architectures. Check which is your GPU's architecture [here](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)

The RGBiD-SLAM algorithm initialises two independent streams in GPU (one for the  camera tracking front-end, and one for the loop closing back-end). This functionality is only available in cuda toolkit v7.0 or later (see [this link](https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/))

Setup of NVIDIA GPUs and enabling of CUDA in Ubuntu systems is not trivial (specially in a laptop). Before going further plase make sure you have followed the steps described in INFO_CUDA_INSTALLATION.txt 

Install the following packages through synaptic software manager

ros-indigo-desktop

ros-indigo-openni2-launch

libsuitesparse-dev 

pcl-1.7-all-dev

libboost-dev

If you are using the Asus Xtion Pro Live camera you have execute this in console:

	sudo gedit /etc/openni/GlobalDefaults.ini

Then find the line that says ";UsbInterface=2" and uncomment it. Save the file.


#Compilation

Download the repository in a zip file

First we compile the external dependencies included in this distribution:

  	cd ~/RGBiD-SLAM/ThirdParty
  	
  	chmod +x build_dir.sh build_all_dirs.sh
  	
  	./build_all_dirs.sh  
  
 This code uses variables dependent on the CUDA architecture of your NVIDIA GPU for optimal performance, unfourtunately it is not posible to define them automatically at compile time for the bridge functions implemented in host (CPU). We have to do it manually before compiling:
 
 	gedit ~/RGBiD-SLAM/src/cuda/device.hpp 
 	
 change the global variables (those preceded by #define) depending on the CUDA architecture of your NVIDIA GPU
  
Now we proceed to compile the program:

	cd ~/RGBiD-SLAM/
	
	mkdir build
	
	cd build
	
	cmake ../
	
	make -j4
  
Unzip ~/RGBiD-SLAM/data/ORBvoc.yml.zip

Then copy configuration files from ~/RGBiD-SLAM/config_data into ~/RGBiD-SLAM/build/tools/

#Usage


Plug your Asus Xtion Pro Live and do

In one terminal tab:

      roscore
  
In other terminal tab:

      roslaunch openni2_launch openni2.launch color_depth_synchronization:=true depth_registration:=true

  
In other terminal tab (dont forget to write 'optirun' first if executing the app on a laptop):  

        cd ~/RGBiD-SLAM/build/tools/    
        {optirun} ./RGBID_SLAMapp -config visodoRGBDconfig.ini -calib calibration_factory.ini  
    
You can run the SLAM app on a previously recorded sequence of images. This sequence must be saved in the same format as sequences in the TUM benchmark dataset. Verify first that the RGB and depth images are synchronised checking the rgb.txt and depth.txt files within the dataset folder. 
      	
      	./RGBID_SLAMapp  -eval folder_where_the_sequence_is -config visodoRGBDconfig.ini -calib calibration_factory.ini

Note that you can calibrate your RGB-D camera and provide your own calibration file for greater accuracy
  


This library includes also some utility applications: 
 

        ./openni_grabber my_folder_name {0,1} 
  
creates my_folder_name where an RGB-D stream is recorded in a sequence of images following the TUM benchmark format (to be able to record remember to follow steps in point 4 before)
0(default) -> record all images  from the stream;  1 -> take snapshots (press 's' to take one)


  
 	./openni_rgb_depth_ir_shot my_folder_name	
	
This is a simple program to get RGB, depth and IR images mainly for
calibration of RGB-D cameras. Upon pressing 's' a shot of each of the 
3 streams is saved. 

It is not possible to obtain synchronised RGB and IR streams, and thus the
shots of each stream are taken consecutively. As a consequence, to obtain images 
for calibration, one should design an experimental setup where the 
pattern and the camera can be rigidly fixed before taking one capture.

Note that in order to calibrate the depth(IR) camera as well as the stereo transofrmation btw. RGB and depth(IR) cameras you have to set "depth_registration:=false" in the "roslaunch" command.
   
 ###Disclaimer
 
 There is the option of disabling registration also when running the SLAM app an use a customised calibration 
 for the complete RGB-D sensor (RGB intrinsics, IR intrinsics, RGB-IR stereo, depth distortion). 
 However for the moment I dont recommend it since I am still working on the calibration model and the app might not run propperly.
 I recommend using always depth_registration:=true and provide only the RGB focals and principal point in the calibration file.
 
	







