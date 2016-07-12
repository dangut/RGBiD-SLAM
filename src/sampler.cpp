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

#include "sampler.h"

Sampler::Sampler()
{  
  m_rho_exhausted = true;
};

Sampler::Sampler(int init)
{
  m_rng.seed(init);
  m_rho_exhausted = true;
};

void Sampler::reset(int init)
{
  m_rng.seed(init);
  m_rho_exhausted = true;
}

Sampler::~Sampler()
{}

double Sampler::dummy()
{ return 0.5;}

//Generate random number from uniform distribution with Mersenne Twister in [0,1) )
double Sampler::rand_uniform01()
{
   double num = static_cast<double> (m_rng() - m_rng.min());
   double den = static_cast<double> (m_rng.max()- m_rng.min());
   return num/(den+1.0);
}

//Generate random number from uniform distribution with Mersenne Twister in [0,1] )
double Sampler::rand_uniform0o1o()
{
   double num = static_cast<double> (m_rng() - m_rng.min());
   double den = static_cast<double> (m_rng.max()- m_rng.min());
   return num/den;
}
      
//Generate random gaussian value from uniform distribution (Box-Muller algorithm)
double Sampler::rand_gaussian(double mean, double sigma)
{
    //Same rho can be used to generate 2 independent gaussian samples (when the 2 samples are generated, rho is 'exhausted' -> generate new one)
    const double pi = 3.14159265358979323846;

    if(m_rho_exhausted)
    {
      m_r1 = 2*pi*rand_uniform01();
      m_r2 = rand_uniform01();
      m_rho = sqrt(-2 * log(1-m_r2));
      m_rho_exhausted = false;
    } 
    else 
    {
      m_rho_exhausted = true;
    }
      
    return m_rho * (m_rho_exhausted ? cos(m_r1) : sin(m_r1)) * sigma + mean;     
}

//Generate uniform random sample in unit sphere
void Sampler::rand_uniform_unit_sphere(double& x, double& y)
{ 
    const double pi = 3.14159265358979323846;
    
    double th = 2*pi*rand_uniform01();
    double u = rand_uniform0o1o() + rand_uniform0o1o();
      
    double r = ((u>1) ? 2-u : u);
    
    x = r*cos(th); 
    y = r*sin(th);    
    
    return;
}
