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
* Copyright (C) 2006 Pedro Felzenszwalb
* 
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/


#ifndef SEGMENTERGRAPH_HPP_
#define SEGMENTERGRAPH_HPP_

#include <vector>


namespace RGBID_SLAM 
{  
  namespace Segmentation
  {

    struct Edge
    {
      int ini_;
      int end_;
      float w_; 
      
      bool operator< (const Edge& other) const  { return w_ < other.w_; } 
    };
    
    
    struct Vertex
    {
      Vertex(int i, float th)
      {
        parent_ = i;
        rank_ = 0;
        size_ = 1;
        th_ = th;
      }
      
      int parent_;
      int rank_;
      int size_;
      float th_;
    };


    class Graph
    {
      public:
        Graph (float kTh, int min_segment_size, int Nvertices);
        
        void applySegmentation(std::vector<Edge>& edges);
        
        int findParent(int id);
      
        std::vector<Vertex> vertices_; 
        
      private:
      
        int join(int id1, int id2);
        
        float kTh_;
        int min_segment_size_;
    };
    
  }
}

#endif
