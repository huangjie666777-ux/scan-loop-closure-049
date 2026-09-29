#include "icp3d/se3.h"

#include <cmath>

namespace icp3d {

namespace {
Eigen::Matrix3d hat(const Eigen::Vector3d &v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

// Rodrigues：由旋转向量返回旋转矩阵。
Eigen::Matrix3d expRotation(const Eigen::Vector3d &w) {
  const double angle = w.norm();
  if (angle < 1e-12) {
    return Eigen::Matrix3d::Identity() + hat(w) + 0.5 * hat(w) * hat(w);
  }
  const Eigen::Vector3d axis = w / angle;
  const double s = std::sin(angle);
  const double c = std::cos(angle);
  return c * Eigen::Matrix3d::Identity() + (1.0 - c) * axis * axis.transpose() +
         s * hat(axis);
}
} // namespace

RigidTransform inverse(const RigidTransform &transform) {
  RigidTransform result;
  result.rotation = transform.rotation.transpose();
  result.translation = -result.rotation * transform.translation;
  return result;
}

RigidTransform multiply(const RigidTransform &a, const RigidTransform &b) {
  RigidTransform result;
  result.rotation = a.rotation * b.rotation;
  result.translation = a.rotation * b.translation + a.translation;
  return result;
}

RigidTransform se3Exp(const Twist &twist) {
  const Eigen::Vector3d w = twist.head<3>();
  const Eigen::Vector3d v = twist.tail<3>();
  const double angle = w.norm();
  RigidTransform result;
  result.rotation = expRotation(w);
  if (angle < 1e-12) {
    // V = I + 1/2 hat(w) + 1/6 hat(w)^2
    const Eigen::Matrix3d hw = hat(w);
    const Eigen::Matrix3d vmat =
        Eigen::Matrix3d::Identity() + 0.5 * hw + (1.0 / 6.0) * hw * hw;
    result.translation = vmat * v;
  } else {
    const Eigen::Matrix3d hw = hat(w);
    const Eigen::Matrix3d vmat =
        Eigen::Matrix3d::Identity() +
        (1.0 - std::cos(angle)) / (angle * angle) * hw +
        (angle - std::sin(angle)) / (angle * angle * angle) * hw * hw;
        result.translation = vmat * v;
  }
  return result;
}

Twist se3Log(const RigidTransform &transform) {
  Twist twist = Twist::Zero();
  const Eigen::Matrix3d &r = transform.rotation;
  const double cosAngle =
      std::max(-1.0, std::min(1.0, (r.trace() - 1.0) * 0.5));
  const double angle = std::acos(cosAngle);
  Eigen::Vector3d w = Eigen::Vector3d::Zero();
  Eigen::Matrix3d vinv = Eigen::Matrix3d::Identity();
  if (angle > 1e-9) {
    const double factor = angle / (2.0 * std::sin(angle));
    Eigen::Matrix3d skew = factor * (r - r.transpose());
    w << skew(2, 1), skew(0, 2), skew(1, 0);
    const Eigen::Matrix3d hw = hat(w);
    // V^{-1} = I - 1/2 hat(w) + (1/theta^2)(1 - theta/(2 tan(theta/2))) hat(w)^2
    const double half = 0.5 * angle;
    const double c2 =
        (1.0 - half / std::tan(half)) / (angle * angle);
    vinv = Eigen::Matrix3d::Identity() - 0.5 * hw + c2 * hw * hw;
  } else {
    // 小角展开，保证数值稳定。
    const Eigen::Matrix3d skew = 0.5 * (r - r.transpose());
    w << skew(2, 1), skew(0, 2), skew(1, 0);
    const Eigen::Matrix3d hw = hat(w);
    vinv = Eigen::Matrix3d::Identity() - 0.5 * hw;
  }
  twist.head<3>() = w;
  twist.tail<3>() = vinv * transform.translation;
  return twist;
}

Matrix6d se3LeftJacobian(const Eigen::Vector3d &w) {
  Matrix6d j = Matrix6d::Identity();
  const double angle = w.norm();
  const Eigen::Matrix3d hw = hat(w);
  Eigen::Matrix3d jr3;
  if (angle < 1e-9) {
    jr3 = Eigen::Matrix3d::Identity() + 0.5 * hw + (1.0 / 6.0) * hw * hw;
  } else {
    jr3 = Eigen::Matrix3d::Identity() +
          (1.0 - std::cos(angle)) / (angle * angle) * hw +
          (angle - std::sin(angle)) / (angle * angle * angle) * hw * hw;
  }
  // 左雅可比分块（扭量顺序 [w; v]）：[[J_r3, 0],[Q, J_r3]]
  Eigen::Matrix3d q = Eigen::Matrix3d::Zero();
  if (angle >= 1e-9) {
    const double t = angle;
    const double s = std::sin(t);
    const double c = std::cos(t);
    q = 0.5 * hat(w) +
        (1.0 / (t * t) - (1.0 + c) / (2.0 * t * s)) * hw * hw +
        0.5 * ((1.0 + c) / (t * s) - 2.0 / (t * t)) * (hw * hw.transpose() +
                                                        hw.transpose() * hw);
  } else {
    q = 0.5 * hw;
  }
  j.topLeftCorner<3, 3>() = jr3;
  j.bottomRightCorner<3, 3>() = jr3;
  j.bottomLeftCorner<3, 3>() = q;
  return j;
}

Matrix6d se3LeftJacobianInverse(const Eigen::Vector3d &w) {
  Matrix6d result = Matrix6d::Identity();
  const double angle = w.norm();
  const Eigen::Matrix3d hw = hat(w);
  Eigen::Matrix3d jrinv;
  if (angle < 1e-9) {
    jrinv = Eigen::Matrix3d::Identity() - 0.5 * hw + (1.0 / 12.0) * hw * hw;
  } else {
    const double half = 0.5 * angle;
    jrinv = Eigen::Matrix3d::Identity() - 0.5 * hw +
            (1.0 - half / std::tan(half)) / (angle * angle) * hw * hw;
  }
  result.topLeftCorner<3, 3>() = jrinv;
  result.bottomRightCorner<3, 3>() = jrinv;
  // 左下块 -J^{-1} Q J^{-1}，Q 由 J - blockdiag 提取。
  const Matrix6d j = se3LeftJacobian(w);
  const Eigen::Matrix3d q = j.bottomLeftCorner<3, 3>();
  result.bottomLeftCorner<3, 3>() = -jrinv * q * jrinv;
  return result;
}

Matrix6d adjoint(const RigidTransform &transform) {
  Matrix6d ad = Matrix6d::Zero();
  ad.topLeftCorner<3, 3>() = transform.rotation;
  ad.bottomRightCorner<3, 3>() = transform.rotation;
  ad.bottomLeftCorner<3, 3>() =
      hat(transform.translation) * transform.rotation;
  return ad;
}

} // namespace icp3d
