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

#include <iostream>
#include <algorithm>
#include "graph_segmenter.h"


RGBID_SLAM::Segmentation::Graph::Graph(float kTh, int min_segment_size, int Nvertices)
{
  kTh_ = kTh;
  min_segment_size_ = min_segment_size;
  
  vertices_.reserve(Nvertices);
  
  for (int i=0; i< Nvertices; i++)
  {
    Vertex v(i,kTh_);
    
    vertices_.push_back(v);
  }
};
      
    
int RGBID_SLAM::Segmentation::Graph::join(int id1, int id2)
{
  if (vertices_[id1].rank_ > vertices_[id2].rank_)
  {
    vertices_[id1].size_ += vertices_[id2].size_;
    vertices_[id2].parent_ = vertices_[id1].parent_;
  }
  else
  {
    vertices_[id2].size_ += vertices_[id1].size_;
    vertices_[id1].parent_ = vertices_[id2].parent_;
    
    if (vertices_[id1].rank_ == vertices_[id2].rank_)
    {
      vertices_[id2].rank_++;
    }
  }
  
  return vertices_[id2].parent_;
}

int RGBID_SLAM::Segmentation::Graph::findParent(int id)
{
  int pid = id;
  
  while (pid != vertices_[pid].parent_)
  {
    pid = vertices_[pid].parent_;
  }
  
  vertices_[id].parent_ = pid;
  
  return pid;
}

void 
RGBID_SLAM::Segmentation::Graph::applySegmentation(std::vector<Edge>& edges)
{  
  std::sort(edges.begin(), edges.end());
  
  for (int i=0; i<edges.size(); i++)
  {
    int id1p = findParent(edges[i].ini_);
    int id2p = findParent(edges[i].end_);
    
    if (id1p != id2p)
    {
      if ((edges[i].w_ <= (vertices_[id1p].th_)) && (edges[i].w_ <= (vertices_[id2p].th_)))
      {
        int new_pid = join(id1p,id2p);
        vertices_[new_pid].th_ = edges[i].w_+ (kTh_ / ((float) vertices_[new_pid].size_));        
      }
    }
  }
  
  for (int i=0; i<edges.size(); i++)
  {
    int id1p = findParent(edges[i].ini_);
    int id2p = findParent(edges[i].end_);
    
    if (id1p != id2p)
    {
      if ((vertices_[id1p].size_ < min_segment_size_) || (vertices_[id2p].size_ < min_segment_size_))
      {
        join(id1p,id2p);  
      }
    }  
  }  
}
