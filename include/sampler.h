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

#ifndef SAMPLER_H
#define SAMPLER_H

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>


class Sampler{

 public:

  Sampler();
  Sampler(int init);
  ~Sampler();
  
  void reset(int init);
  
  double dummy();
  double rand_gaussian(double mean, double sigma);
  double rand_uniform01();
  
  double rand_uniform0o1o();
  void rand_uniform_unit_sphere(double& x, double& y);
  
  double m_dummy;
  boost::mt19937 m_rng;
  
 private:
  double m_r1, m_r2, m_rho;
  bool m_rho_exhausted;    
};

#endif
