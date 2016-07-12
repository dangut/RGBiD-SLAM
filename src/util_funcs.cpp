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

#include <Eigen/SVD>
#include <Eigen/Cholesky>
//#include <Eigen/Geometry>
//#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "util_funcs.h"

namespace RGBID_SLAM 
{  
    
  Vector6ft logMap(const Matrix3ft& matrix, const Vector3ft& trans)
  {
    Eigen::JacobiSVD<Matrix3ft> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
    Matrix3ft R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);
    double theta2 = theta*theta;
    double th_by_sinth;
      
    if ( s < 1e-5 )
    {
      th_by_sinth = 1.0 + (1.0/6.0)*theta2 + (7.0/360.0)*theta2*theta2;
    }
    else
    {
      th_by_sinth = theta/s;
    }
    
    double vth = th_by_sinth / 2.0;
    rx *= vth; ry *= vth; rz *= vth;
    
    Eigen::Vector3d omega_t = Eigen::Vector3d(rx, ry, rz);
    Eigen::Matrix3d Omega_t = skew(omega_t);
    Eigen::Matrix3d Omega_t2 = Omega_t*Omega_t;
    double th = omega_t.norm();
    Eigen::Matrix3d Q;
    Eigen::Matrix3d Qinv;
    
    if (th<0.00001)
      Q = (Eigen::Matrix3d::Identity() + (1.0/2.0)*Omega_t + (1.0/6.0)*Omega_t2);
    else
      Q = (Eigen::Matrix3d::Identity()
          + (1-cos(theta))/(theta*theta)*Omega_t
          + (1-(sin(theta)/theta))/(theta*theta)*Omega_t2);
          
    Qinv = Q.inverse();
    Eigen::Vector3d v_t = Qinv*(trans.cast<double>());
    
    Vector6ft twist;
    twist.block(0,0,3,1) = v_t.cast<float_trafos>();
    twist.block(3,0,3,1) = omega_t.cast<float_trafos>();
    
    return twist;
  }
  
  ///////////////////////////////
  Matrix4ft expMap(const Vector3ft& omega_tf, const Vector3ft& v_tf)
  {
    //const Eigen::Vector3d v_t = Eigen::Map<Eigen::Vector3d>(v_tf).cast<double>();
    Eigen::Vector3d omega_t = omega_tf.cast<double>();
    Eigen::Vector3d v_t = v_tf.cast<double>();
    double theta = omega_t.norm();

    Eigen::Matrix3d Omega_t = skew(omega_t);

    Eigen::Matrix3d R;
    Eigen::Matrix3d Q;
    Matrix4ft Trafo = Matrix4ft::Identity();
    Eigen::Matrix3d Omega_t2 = Omega_t*Omega_t;

    if (theta<0.00001)
    {        
      R = (Eigen::Matrix3d::Identity() + Omega_t + (1.0/2.0)*Omega_t2);
      Q = (Eigen::Matrix3d::Identity() + (1.0/2.0)*Omega_t + (1.0/6.0)*Omega_t2);
    }
    else
    {
      R = (Eigen::Matrix3d::Identity()
      + sin(theta)/theta *Omega_t
      + (1-cos(theta))/(theta*theta)*Omega_t2);

      Q = (Eigen::Matrix3d::Identity()
      + (1-cos(theta))/(theta*theta)*Omega_t
      + (1-(sin(theta)/theta))/(theta*theta)*Omega_t2);
    }
    
    Matrix3ft Rf = forceOrthogonalisation(R.cast<float_trafos>());
    
    //Eigen::Matrix3ft Rf = (R.cast<float>());
    
    Trafo.block<3,3>(0,0) = Rf;
    Trafo.block<3,1>(0,3) = Q.cast<float_trafos>()*v_t.cast<float_trafos>();

    return Trafo;
  } 

  Matrix3ft expMapRot(const Vector3ft& omega_tf)
  {
    //const Eigen::Vector3d v_t = Eigen::Map<Eigen::Vector3d>(v_tf).cast<double>();
    Eigen::Vector3d omega_t = omega_tf.cast<double>();
    double theta = omega_t.norm();

    Eigen::Matrix3d Omega_t = skew(omega_t);

    Eigen::Matrix3d R;
    Eigen::Matrix3d Omega_t2 = Omega_t*Omega_t;

    if (theta<0.00001)
      R = ( Eigen::Matrix3d::Identity() + Omega_t + (1.0/2.0)*Omega_t2  ); 
    else
      R = ( Eigen::Matrix3d::Identity()
            + sin(theta)/theta *Omega_t
            + (1-cos(theta))/(theta*theta)*Omega_t2 );        
    
    Matrix3ft Rf = forceOrthogonalisation(R.cast<float_trafos>());        
    
    //Eigen::Matrix3ft Rf = (R.cast<float>());
    
    return Rf;
  }
  
  Matrix3ft forceOrthogonalisation(const Matrix3ft& matrix)
  {
    Eigen::JacobiSVD<Matrix3ft> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
    Matrix3ft R = svd.matrixU() * svd.matrixV().transpose();
    return R;
  }
  
  void goldenSectionSpiral ( std::vector<Point3D>& sphere_points, int Npoints)
  {
    sphere_points.clear();
    float inc = 3.141592f*(3.f-sqrt(5.f));
    float off = 2.f / ((float) Npoints);

    for(int i=0; i < Npoints; i++)
    {
      float y = ((float) i)*off - 1.f + off/2.f;
      float r = sqrt(1-y*y);
      float phi = ((float) i)*inc;
      Point3D point_i;
      point_i.x = cos(phi)*r; point_i.y = y; point_i.z = sin(phi)*r;
      //std::cout << "Point 3d: " << point_i.x << point_i.y << point_i.z << std::endl;
      sphere_points.push_back(point_i);
    }
  }
  
  void computeBresenhamSemicircle(std::vector<int> &umax, int radius)
  {
    umax.resize(radius + 2);

    int v, v0, vmax = (int) (radius * std::sqrt(2.f) / 2 + 1.f);
    int vmin = ((int) (radius * std::sqrt(2.f) / 2)) +1;
    //Build Bresenham circle in one octant
    for (v = 0; v <= vmax; ++v)
      umax[v] = (int) (std::sqrt((double)radius * radius - v * v)+0.5);

    // Make sure we are symmetric (mirror for whole quadrant)
    for (v = radius, v0 = 0; v >= vmin; --v)
    {
      while (umax[v0] == umax[v0 + 1])
          ++v0;          
      umax[v] = v0;
      ++v0;
    }
  }
  

  bool compare_keypoints(const cv::KeyPoint& p1, const cv::KeyPoint& p2) 
  {
    return p1.response > p2.response;
  }   
  
  void descVec2DescMat(std::vector<cv::Mat> &desc_vec, cv::Mat &desc_mat) 
  {
    desc_mat = cv::Mat::zeros(desc_vec.size(), 32, CV_8UC1);
    
    //std::cout << "desc_vecsize: " << desc_vec.size() << std::endl;

    for (int i = 0; i < desc_mat.rows; i++)
    {
      desc_vec[i].copyTo(desc_mat.row(i));
    }
  }
  
  void alignPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud, Eigen::Matrix3d R_WC, Eigen::Vector3d t_WC)
  {
    for (pcl::PointCloud<pcl::PointXYZRGB>::iterator it = point_cloud->begin();
          it != point_cloud->end(); it++)
    {
      Eigen::Vector3d Xcam;
      Xcam << (*it).x, (*it).y, (*it).z;
      Eigen::Vector3d Xworld = R_WC*Xcam + t_WC;
      (*it).x = Xworld[0];
      (*it).y  = Xworld[1];
      (*it).z  = Xworld[2];
    }
  }
  
  void 
  copyPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_src, pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_dst)
  { 
    point_cloud_dst->clear();
    point_cloud_dst->reserve(point_cloud_src->size());
    
    for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator it = point_cloud_src->begin();
          it != point_cloud_src->end(); it++)
    {
      point_cloud_dst->push_back(*it);
    }
    
  } 
  
  void 
  copyNormalCloud(pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_src, pcl::PointCloud<pcl::Normal>::Ptr normal_cloud_dst)
  {       
    normal_cloud_dst->clear();
    normal_cloud_dst->reserve(normal_cloud_src->size()); 
    
    for (pcl::PointCloud<pcl::Normal>::const_iterator it = normal_cloud_src->begin();
          it != normal_cloud_src->end(); it++)
    {
      normal_cloud_dst->push_back(*it);
    }
    
  } 
  
  double
  computeConvexHullArea(const std::vector<cv::Point2f> &points)
  {
    std::vector<cv::Point2f> ch;
    cv::convexHull(cv::Mat(points), ch, false);
    //std::cout << "        points size: " << points.size() << std::endl;
    double area = 0.0;
    for (int i=0; i<ch.size(); i++)
    {
      int ip1 = (i==(ch.size()-1)) ? 0 : i+1;
      double dx = ch[ip1].x-ch[i].x;
      double y = (ch[ip1].y+ch[i].y) / 2.f;
      area += dx*y;
      //std::cout << "                     dx: " << dx << std::endl;
      //std::cout << "                     y: " << y << std::endl;
    }
    //std::cout << "                     Area: " << area << std::endl;
    area = std::abs(area);
    return area;
  }    
}
