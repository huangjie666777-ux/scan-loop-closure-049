#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace icp3d {

// 将一个齐次刚体变换分解为旋转和平移。
struct RigidTransform {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

// 校验旋转：有限、正交（R^T R = I）且 det(R) = 1，不接受缩放或镜像。
// 数值容差：|R^T R - I| 的逐元素误差与 |det(R) - 1| 均需 <= tolerance。
void validateRotation(const Eigen::Matrix3d &rotation, double tolerance = 1e-9);

RigidTransform fromMatrix4(const Eigen::Matrix4d &transform,
                           double tolerance = 1e-9);
Eigen::Matrix4d toMatrix4(const RigidTransform &transform);

// 累积增量：pose = increment * pose（均为源->目标约定）。
RigidTransform compose(const RigidTransform &increment,
                       const RigidTransform &pose);

// 旋转角（弧度），用于旋转收敛判据。
double rotationAngle(const Eigen::Matrix3d &rotation);

struct Correspondence {
  Eigen::Vector3d source;
  Eigen::Vector3d target;
};

// 由保留配对求最小二乘刚体增量（Umeyama/Horn，SVD 求解），无缩放。
// 配对少于 3 组，或去质心后点集共线（sigma2 <=
// rankEpsilon * max(1, sigma1)）时抛出 std::runtime_error。
// 非共线平面点（协方差秩为 2）仍可唯一确定刚体旋转。
RigidTransform estimateRigid(const std::vector<Correspondence> &pairs,
                             double rankEpsilon = 1e-10);

} // namespace icp3d
