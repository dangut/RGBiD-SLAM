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

#ifndef G2O_TRAFOSO3R3_H_
#define G2O_TRAFOSO3R3_H_

#include "so3_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {

  typedef Eigen::Matrix<double, 6, 1, Eigen::ColMajor> Vector6d;
  typedef Eigen::Matrix<double, 7, 1, Eigen::ColMajor> Vector7d;

  class G2O_TYPES_POSEGRAPH_API TrafoSO3R3 {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    protected:

      Eigen::Quaterniond _r;
      Vector3D _t;

    public:
      TrafoSO3R3(){
        _r.setIdentity();
        _t.setZero();
      }

      TrafoSO3R3(const Matrix3D& R, const Vector3D& t):_r(Eigen::Quaterniond(R)),_t(t){ 
        normalizeRotation();
      }

      TrafoSO3R3(const Eigen::Quaterniond& q, const Vector3D& t):_r(q),_t(t){
        normalizeRotation();
      }

      /**
       * templaized constructor which allows v to be an arbitrary Eigen Vector type, e.g., Vector6d or Map<Vector6d>
       */
      template <typename Derived>
        explicit TrafoSO3R3(const Eigen::MatrixBase<Derived>& v)
        {
          assert((v.size() == 6 || v.size() == 7) && "Vector dimension does not match");
          if (v.size() == 6) {
            for (int i=0; i<3; i++){
              _t[i]=v[i];
              _r.coeffs()(i)=v[i+3];
            }
            _r.w() = 0.; // recover the positive w
            if (_r.norm()>1.){
              _r.normalize();
            } else {
              double w2=1.-_r.squaredNorm();
              _r.w()= (w2<0.) ? 0. : sqrt(w2);
            }
          }
          else if (v.size() == 7) {
            int idx = 0;
            for (int i=0; i<3; ++i, ++idx)
              _t(i) = v(idx);
            for (int i=0; i<4; ++i, ++idx)
              _r.coeffs()(i) = v(idx);
            normalizeRotation();
          }
        }

      inline const Vector3D& translation() const {return _t;}

      inline void setTranslation(const Vector3D& t_) {_t = t_;}

      inline const Matrix3D rotation() const 
      {
        Matrix3D R = _r.toRotationMatrix();
        return R;
      }

      void setRotation(const Eigen::Quaterniond& r_) {_r=r_;}

      inline TrafoSO3R3 operator* (const TrafoSO3R3& tr2) const{
        TrafoSO3R3 result(*this);
        result._t += _r*tr2._t;
        result._r*=tr2._r;
        result.normalizeRotation();
        return result;
      }

      inline TrafoSO3R3& operator*= (const TrafoSO3R3& tr2){
        _t+=_r*tr2._t;
        _r*=tr2._r;
        normalizeRotation();
        return *this;
      }

      inline Vector3D operator* (const Vector3D& v) const {
        return _t+_r*v;
      }

      inline TrafoSO3R3 inverse() const{
        TrafoSO3R3 ret;
        ret._r=_r.conjugate();
        ret._t=ret._r*(_t*-1.);
        return ret;
      }

      inline double operator [](int i) const {
        assert(i<7);
        if (i<3)
          return _t[i];
        return _r.coeffs()[i-3];
      }


      inline Vector7d toVector() const{
        Vector7d v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=_r.x();
        v[4]=_r.y();
        v[5]=_r.z();
        v[6]=_r.w();
        return v;
      }

      inline void fromVector(const Vector7d& v){
        _r=Eigen::Quaterniond(v[6], v[3], v[4], v[5]);
        _t=Vector3D(v[0], v[1], v[2]);
      }

      inline Vector6d toMinimalVector() const{
        Vector6d v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=_r.x();
        v[4]=_r.y();
        v[5]=_r.z();
        return v;
      }

      inline void fromMinimalVector(const Vector6d& v){
        double w = 1.-v[3]*v[3]-v[4]*v[4]-v[5]*v[5];
        if (w>0){
          _r=Eigen::Quaterniond(sqrt(w), v[3], v[4], v[5]);
        } else {
          _r=Eigen::Quaterniond(0, -v[3], -v[4], -v[5]);
        }
        _t=Vector3D(v[0], v[1], v[2]);
      }



      Vector6d log() const {
        Vector6d res;
        Matrix3D _R = _r.toRotationMatrix();
        double d =  0.5*(_R(0,0)+_R(1,1)+_R(2,2)-1);
        Vector3D omega;
        Vector3D upsilon;
        
        Vector3D dR = deltaR(_R);

        if (d>0.99999)
        {
          omega=0.5*dR;
        }//TODO: missing special case when theta ~= pi (unlikely but should be considered)
        else
        {
          double theta = acos(d);
          omega = theta/(2*sqrt(1-d*d))*dR;
        }

        upsilon = _t;
        for (int i=0; i<3;i++){
          res[i]=upsilon[i];
        }
        for (int i=0; i<3;i++){
          res[i+3]=omega[i];
        }

        return res;

      }

      Vector3D map(const Vector3D & xyz) const
      {
        return _r*xyz + _t;
      }


      static TrafoSO3R3 exp(const Vector6d & update)
      {
        Vector3D omega;
        for (int i=0; i<3; i++)
          omega[i]=update[i+3];
        Vector3D upsilon;
        for (int i=0; i<3; i++)
          upsilon[i]=update[i];

        double theta = omega.norm();
        Matrix3D Omega = skew(omega);

        Matrix3D R;
        Matrix3D V;
        if (theta<0.00001)
        {
          R = (Matrix3D::Identity() + Omega + 0.5f*Omega*Omega);
        }
        else
        {
          Matrix3D Omega2 = Omega*Omega;

          R = (Matrix3D::Identity()
              + sin(theta)/theta *Omega
              + (1-cos(theta))/(theta*theta)*Omega2);
        }
        return TrafoSO3R3(Eigen::Quaterniond(R),upsilon);
      }

      Eigen::Matrix<double, 6, 6, Eigen::ColMajor> adj() const
      {
        Matrix3D R = _r.toRotationMatrix();
        Eigen::Matrix<double, 6, 6, Eigen::ColMajor> res;
        res.block(0,0,3,3) = R;
        res.block(3,3,3,3) = R;
        res.block(0,3,3,3) = skew(_t)*R;
        res.block(3,0,3,3) = Matrix3D::Zero(3,3);
        return res;
      }

      Eigen::Matrix<double,4,4,Eigen::ColMajor> to_homogeneous_matrix() const
      {
        Eigen::Matrix<double,4,4,Eigen::ColMajor> homogeneous_matrix;
        homogeneous_matrix.setIdentity();
        homogeneous_matrix.block(0,0,3,3) = _r.toRotationMatrix();
        homogeneous_matrix.col(3).head(3) = translation();

        return homogeneous_matrix;
      }

      void normalizeRotation(){
        if (_r.w()<0){
          _r.coeffs() *= -1;
        }
        _r.normalize();
      }

      /**
       * cast SE3Quat into an Isometry3D
       */
      operator Isometry3D() const
      {
        Isometry3D result = (Isometry3D) rotation();
        result.translation() = translation();
        return result;
      }
  };

  inline std::ostream& operator <<(std::ostream& out_str, const TrafoSO3R3& se3)
  {
    out_str << se3.to_homogeneous_matrix()  << std::endl;
    return out_str;
  }

  //G2O_TYPES_SLAM3D_API Eigen::Quaterniond euler_to_quat(double yaw, double pitch, double roll);
  //G2O_TYPES_SLAM3D_API void quat_to_euler(const Eigen::Quaterniond& q, double& yaw, double& pitch, double& roll);
  //G2O_TYPES_SLAM3D_API void jac_quat3_euler3(Eigen::Matrix<double, 6, 6, Eigen::ColMajor>& J, const SE3Quat& t);

} // end namespace

#endif
