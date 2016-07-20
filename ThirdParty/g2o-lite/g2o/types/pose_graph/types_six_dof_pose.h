// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_SIX_DOF_TYPES_POSE
#define G2O_SIX_DOF_TYPES_POSE

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/pose_graph/so3_utils.h"
#include <Eigen/Geometry>
#include "trafo_so3r3.h"

namespace g2o {
namespace types_six_dof_pose {
void init();
}

typedef Eigen::Matrix<double, 6, 6, Eigen::ColMajor> Matrix6d;



/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */
class G2O_TYPES_POSEGRAPH_API VertexSO3R3 : public BaseVertex<6, TrafoSO3R3>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSO3R3();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  virtual void setToOriginImpl() {
    _estimate = TrafoSO3R3();
  }

  virtual void oplusImpl(const double* update_)  {
    Eigen::Map<const Vector6d> update(update_);
    TrafoSO3R3 increment = TrafoSO3R3::exp(update);
    setEstimate(estimate()*increment);
  }
};


/**
 * \brief 6D edge between two Vertex6
 */
class G2O_TYPES_POSEGRAPH_API EdgeSO3R3 : public BaseBinaryEdge<6, TrafoSO3R3, VertexSO3R3, VertexSO3R3>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      EdgeSO3R3();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    //e+de = Log(exp(dx_ij)*iTj*(wTj*exp(dx_j))^-1*wTi*exp(dx_i))
    void computeError()  {
      const VertexSO3R3* vi = static_cast<const VertexSO3R3*>(_vertices[0]);
      const VertexSO3R3* vj = static_cast<const VertexSO3R3*>(_vertices[1]);

      TrafoSO3R3 iTj_measured(_measurement);
      TrafoSO3R3 errorT_= iTj_measured*vj->estimate().inverse()*vi->estimate();
      _error = errorT_.log();
      Eigen::Vector3d error_rot = _error.tail(3);
      Eigen::Matrix3d Q = jacobianR(error_rot);
      Eigen::Matrix3d Qinv = Q.inverse();
      
      Eigen::Matrix<double, 6, 6> de_by_diTj;
      de_by_diTj.setZero();
      
      de_by_diTj.block(0,0,3,3) = Eigen::Matrix3d::Identity();
      de_by_diTj.block(0,3,3,3) = -skew(errorT_.translation());
      de_by_diTj.block(3,3,3,3) = Qinv*errorT_.rotation().transpose();
      
      Eigen::Matrix<double, 6, 6> covariance = de_by_diTj*_proto_information.inverse()*de_by_diTj.transpose();
      _information = covariance.inverse();    
    }

    virtual void linearizeOplus();
    
    virtual void setInformation(const Eigen::Matrix<double,6,6>& proto_information) {  _proto_information =  proto_information;}
    
    Eigen::Matrix<double,6,6> _proto_information; 
   
};


} // end namespace

#endif
