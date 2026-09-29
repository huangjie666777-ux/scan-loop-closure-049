#include "icp3d/rigid.h"

#include <cmath>

namespace icp3d {

namespace {
bool isFiniteMatrix(const Eigen::MatrixXd &m) {
  return m.allFinite();
}
} // namespace

void validateRotation(const Eigen::Matrix3d &rotation, double tolerance) {
  if (!(tolerance > 0.0) || !rotation.allFinite()) {
    throw std::invalid_argument("非法旋转矩阵：必须全部为有限值");
  }
  const double orthoError =
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
          .cwiseAbs()
          .maxCoeff();
  if (orthoError > tolerance) {
    throw std::invalid_argument(
        "非法旋转矩阵：R^T R 不等于 I（含缩放/剪切/非正交列）");
  }
  if (std::abs(rotation.determinant() - 1.0) > tolerance) {
    throw std::invalid_argument("非法旋转矩阵：det(R) 不为 1（拒绝镜像）");
  }
}

RigidTransform fromMatrix4(const Eigen::Matrix4d &transform,
                           double tolerance) {
  if (!transform.allFinite()) {
    throw std::invalid_argument("初始变换包含非有限值");
  }
  RigidTransform result;
  result.rotation = transform.topLeftCorner<3, 3>();
  result.translation = transform.topRightCorner<3, 1>();
  validateRotation(result.rotation, tolerance);
  const double bottomError =
      (transform.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0))
          .cwiseAbs()
          .maxCoeff();
  if (bottomError > tolerance) {
    throw std::invalid_argument("初始变换不是齐次刚体变换（最后一行为 [0 0 0 1]）");
  }
  return result;
}

Eigen::Matrix4d toMatrix4(const RigidTransform &transform) {
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m.topLeftCorner<3, 3>() = transform.rotation;
  m.topRightCorner<3, 1>() = transform.translation;
  return m;
}

RigidTransform compose(const RigidTransform &increment,
                       const RigidTransform &pose) {
  RigidTransform result;
  result.rotation = increment.rotation * pose.rotation;
  result.translation =
      increment.rotation * pose.translation + increment.translation;
  return result;
}

double rotationAngle(const Eigen::Matrix3d &rotation) {
  // theta = acos((tr(R) - 1) / 2)，夹紧到 [-1, 1] 防止数值越界。
  const double cosTheta =
      std::max(-1.0, std::min(1.0, (rotation.trace() - 1.0) * 0.5));
  return std::acos(cosTheta);
}

RigidTransform estimateRigid(const std::vector<Correspondence> &pairs,
                             double rankEpsilon) {
  if (!(rankEpsilon > 0.0)) {
    throw std::invalid_argument("rankEpsilon 必须为正数");
  }
  if (pairs.size() < 3) {
    throw std::runtime_error("有效配对不足 3 组，无法求解刚体变换");
  }

  Eigen::Vector3d sourceMean = Eigen::Vector3d::Zero();
  Eigen::Vector3d targetMean = Eigen::Vector3d::Zero();
  for (const auto &pair : pairs) {
    if (!pair.source.allFinite() || !pair.target.allFinite()) {
      throw std::invalid_argument("配对中存在非有限坐标");
    }
    sourceMean += pair.source;
    targetMean += pair.target;
  }
  const double invN = 1.0 / static_cast<double>(pairs.size());
  sourceMean *= invN;
  targetMean *= invN;

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto &pair : pairs) {
    covariance += (pair.target - targetMean) *
                  (pair.source - sourceMean).transpose();
  }
  covariance *= invN;
  if (!isFiniteMatrix(covariance)) {
    throw std::runtime_error("协方差矩阵含非有限值，无法求解");
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d &s = svd.singularValues();
  const Eigen::Matrix3d &u = svd.matrixU();
  const Eigen::Matrix3d &v = svd.matrixV();
  // 去质心点集秩为 2（非共线三点/平面）即可唯一确定刚体：平面法向
  // 唯一，SVD 仍能给出正确旋转（平面点云不再被误判为退化）。只有秩
  // 不足 2（全部共线）时，绕点列方向的旋转不可观，才判退化。
  const double rankThreshold = rankEpsilon * std::max(1.0, s(0));
  if (s(1) <= rankThreshold) {
    throw std::runtime_error(
        "配对几何退化（点集共线、秩不足 2），无法唯一确定旋转");
  }

  Eigen::Matrix3d rotation = u * v.transpose();
  if (rotation.determinant() < 0.0) {
    // 反射情形：翻转 U 的最后一列，禁止镜像解。
    Eigen::Matrix3d correctedU = u;
    correctedU.col(2) *= -1.0;
    rotation = correctedU * v.transpose();
  }
  RigidTransform result;
  // SVD 的 U、V 本身正交，R 自动满足 R^T R = I；反射情形已在上面修正。
  result.rotation = rotation;
  result.translation = targetMean - result.rotation * sourceMean;
  return result;
}

} // namespace icp3d
