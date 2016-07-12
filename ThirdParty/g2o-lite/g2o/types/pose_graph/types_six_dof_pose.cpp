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

#include "types_six_dof_pose.h"

#include "g2o/core/factory.h"
#include "g2o/stuff/macros.h"

namespace g2o {

using namespace std;
using namespace Eigen;

//G2O_REGISTER_TYPE_GROUP(expmap);
G2O_REGISTER_TYPE(VERTEX_SO3R3, VertexSO3R3);
G2O_REGISTER_TYPE(EDGE_SO3R3, EdgeSO3R3);




VertexSO3R3::VertexSO3R3() : BaseVertex<6, TrafoSO3R3>() {
  setToOriginImpl();
  updateCache();
}

bool VertexSO3R3::read(std::istream& is) {
  Vector7d est;
  for (int i=0; i<7; i++)
    is  >> est[i];
  TrafoSO3R3 wTc;
  wTc.fromVector(est);
  setEstimate(wTc);
  return true;
}

bool VertexSO3R3::write(std::ostream& os) const {
  TrafoSO3R3 wTc(estimate());
  for (int i=0; i<7; i++)
    os << wTc[i] << " ";
  return os.good();
}

EdgeSO3R3::EdgeSO3R3() :
  BaseBinaryEdge<6, TrafoSO3R3, VertexSO3R3, VertexSO3R3>() {
}

bool EdgeSO3R3::read(std::istream& is)  {
  Vector7d meas;
  for (int i=0; i<7; i++)
    is >> meas[i];
  TrafoSO3R3 iTj;
  iTj.fromVector(meas);
  setMeasurement(iTj);
  //TODO: Convert information matrix!!
  for (int i=0; i<6; i++)
    for (int j=i; j<6; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSO3R3::write(std::ostream& os) const {
  TrafoSO3R3 iTj(measurement());
  for (int i=0; i<7; i++)
    os << iTj[i] << " ";
  for (int i=0; i<6; i++)
    for (int j=i; j<6; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}




void EdgeSO3R3::linearizeOplus() {
  
  VertexSO3R3 * vi = static_cast<VertexSO3R3 *>(_vertices[0]);
  TrafoSO3R3 wTi(vi->estimate());

  VertexSO3R3 * vj = static_cast<VertexSO3R3 *>(_vertices[1]);
  TrafoSO3R3 wTj(vj->estimate());

  const TrafoSO3R3 & iTj = _measurement;
  Eigen::Matrix<double,4,4>  iTj_h = iTj.to_homogeneous_matrix();
  
  TrafoSO3R3 iTjTwTi = iTj*wTj.inverse()*wTi;
  Eigen::Matrix<double,4,4>  E = iTjTwTi.to_homogeneous_matrix();
  
  Vector3D tWi_minus_tWj = wTi.translation() - wTj.translation();
  Matrix3D wRj = wTj.rotation();
  Vector3D minus_tji = wRj.transpose()*tWi_minus_tWj;
  //Matrix3D skew_minus_tji = skew(minus_tji);
  
  Vector6d error = iTjTwTi.log();
  Vector3D error_rot = error.tail(3);
  Matrix3D Q = jacobianR(error_rot);
  Matrix3D Qinv = Q.inverse();
  //Matrix3D skew_error_rot = skew(error_rot);
  //Matrix3D cbh_3order_factor = Matrix3D::Identity() - 0.5*skew_error_rot + (1.f/12.f)*skew_error_rot*skew_error_rot;
  
  _jacobianOplusXj.setZero();
  
  _jacobianOplusXj.block(0,0,3,3) = -iTj_h.block(0,0,3,3);
  _jacobianOplusXj.block(0,3,3,3) = iTj_h.block(0,0,3,3)*skew(minus_tji);
  _jacobianOplusXj.block(3,3,3,3) = -Qinv*iTj_h.block(0,0,3,3);
  
  _jacobianOplusXi.setZero();
  _jacobianOplusXi.block(0,0,3,3) = E.block(0,0,3,3);
  _jacobianOplusXi.block(3,3,3,3) = Qinv*E.block(0,0,3,3);
  
}



} // end namespace
