#include "icp3d/se3.h"

#include <cmath>

#include "icp3d/rigid.h"

namespace icp3d {

namespace {

Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return m;
}

Eigen::Vector3d rotationLog(const Eigen::Matrix3d &rotation) {
  const double cosAngle =
      std::max(-1.0, std::min(1.0, (rotation.trace() - 1.0) * 0.5));
  const double angle = std::acos(cosAngle);
  if (angle < 1e-8) {
    const Eigen::Matrix3d approximate = rotation - Eigen::Matrix3d::Identity();
    return Eigen::Vector3d(approximate(2, 1) - approximate(1, 2),
                           approximate(0, 2) - approximate(2, 0),
                           approximate(1, 0) - approximate(0, 1)) * 0.5;
  }
  Eigen::Matrix3d logRotation =
      angle / (2.0 * std::sin(angle)) *
      (rotation - rotation.transpose());
  return Eigen::Vector3d(logRotation(2, 1), logRotation(0, 2),
                         logRotation(1, 0));
}

} // namespace

Se3Pose inverse(const Se3Pose &pose) {
  Se3Pose result;
  result.rotation = pose.rotation.transpose();
  result.translation = -result.rotation * pose.translation;
  return result;
}

Se3Pose multiply(const Se3Pose &left, const Se3Pose &right) {
  Se3Pose result;
  result.rotation = left.rotation * right.rotation;
  result.translation =
      left.rotation * right.translation + left.translation;
  return result;
}

Eigen::Matrix4d poseToMatrix4(const Se3Pose &pose) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.topLeftCorner<3, 3>() = pose.rotation;
  transform.topRightCorner<3, 1>() = pose.translation;
  return transform;
}

Se3Pose poseFromMatrix4(const Eigen::Matrix4d &transform, double tolerance) {
  RigidTransform rigid =
      icp3d::fromMatrix4(transform, tolerance);
  return {rigid.rotation, rigid.translation};
}

Se3Pose expSe3(const Se3Tangent &tangent) {
  const Eigen::Vector3d rho = tangent.head<3>();
  const Eigen::Vector3d phi = tangent.tail<3>();
  const double angle = phi.norm();
  const Eigen::Matrix3d phiHat = skew(phi);
  const Eigen::Matrix3d phiHat2 = phiHat * phiHat;

  Eigen::Matrix3d rotation;
  Eigen::Matrix3d v;
  if (angle < 1e-8) {
    rotation = Eigen::Matrix3d::Identity() + phiHat + 0.5 * phiHat2;
    v = Eigen::Matrix3d::Identity() + 0.5 * phiHat +
        (1.0 / 6.0) * phiHat2;
  } else {
    rotation = Eigen::Matrix3d::Identity() +
               (std::sin(angle) / angle) * phiHat +
               ((1.0 - std::cos(angle)) / (angle * angle)) * phiHat2;
    v = Eigen::Matrix3d::Identity() +
        ((1.0 - std::cos(angle)) / (angle * angle)) * phiHat +
        ((angle - std::sin(angle)) / (angle * angle * angle)) * phiHat2;
  }

  Se3Pose pose;
  pose.rotation = rotation;
  pose.translation = v * rho;
  return pose;
}

Se3Tangent logSe3(const Se3Pose &pose) {
  const Eigen::Vector3d phi = rotationLog(pose.rotation);
  const double angle = phi.norm();
  const Eigen::Matrix3d phiHat = skew(phi);
  const Eigen::Matrix3d phiHat2 = phiHat * phiHat;

  Eigen::Matrix3d vInverse;
  if (angle < 1e-8) {
    vInverse = Eigen::Matrix3d::Identity() - 0.5 * phiHat +
               (1.0 / 12.0) * phiHat2;
  } else {
    const double coefficient =
        1.0 / (angle * angle) -
        1.0 / (2.0 * angle * std::tan(angle * 0.5));
    vInverse = Eigen::Matrix3d::Identity() - 0.5 * phiHat +
               coefficient * phiHat2;
  }

  Se3Tangent tangent;
  tangent.head<3>() = vInverse * pose.translation;
  tangent.tail<3>() = phi;
  return tangent;
}

Se3Tangent relativeLog(const Se3Pose &measurement,
                       const Se3Pose &sourcePose,
                       const Se3Pose &targetPose) {
  const Se3Pose predicted =
      multiply(inverse(targetPose), sourcePose);
  const Se3Pose residualTransform =
      multiply(inverse(measurement), predicted);
  return logSe3(residualTransform);
}

} // namespace icp3d
