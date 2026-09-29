#pragma once

#include <Eigen/Dense>

namespace icp3d {

using Se3Tangent = Eigen::Matrix<double, 6, 1>;

struct Se3Pose {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

Se3Pose inverse(const Se3Pose &pose);
Se3Pose multiply(const Se3Pose &left, const Se3Pose &right);
Eigen::Matrix4d poseToMatrix4(const Se3Pose &pose);
Se3Pose poseFromMatrix4(const Eigen::Matrix4d &transform,
                        double tolerance = 1e-9);

// 切向量顺序为 [translation; axis-angle]。
Se3Pose expSe3(const Se3Tangent &tangent);
Se3Tangent logSe3(const Se3Pose &pose);
Se3Tangent relativeLog(const Se3Pose &measurement,
                       const Se3Pose &sourcePose,
                       const Se3Pose &targetPose);

} // namespace icp3d
