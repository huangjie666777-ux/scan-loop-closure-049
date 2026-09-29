#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <vector>

#include "icp3d/multiscan.h"

using namespace icp3d;

namespace {

Eigen::Matrix4d translationPose(double x, double y, double z) {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.topRightCorner<3, 1>() = Eigen::Vector3d(x, y, z);
  return pose;
}

PointCloud gridPoints() {
  PointCloud points;
  for (int x = 0; x <= 3; ++x) {
    for (int y = 0; y <= 3; ++y) {
      for (int z = 0; z <= 3; ++z) {
        points.emplace_back(static_cast<double>(x), static_cast<double>(y),
                            static_cast<double>(z));
      }
    }
  }
  return points;
}

PointCloud select(const std::vector<int> &indices) {
  const PointCloud all = gridPoints();
  PointCloud selected;
  for (int index : indices) {
    selected.push_back(all[static_cast<std::size_t>(index)]);
  }
  return selected;
}

PointCloud localFromGlobal(const PointCloud &global,
                           const Eigen::Matrix4d &pose) {
  PointCloud local;
  const Eigen::Matrix4d inversePose = pose.inverse();
  for (const Point &point : global) {
    Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
    local.push_back((inversePose * homogeneous).head<3>());
  }
  return local;
}

double poseError(const std::vector<Eigen::Matrix4d> &estimated,
                 const std::vector<Eigen::Matrix4d> &truth) {
  double error = 0.0;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    error += (estimated[i].topRightCorner<3, 1>() -
              truth[i].topRightCorner<3, 1>()).norm();
  }
  return error;
}

double closureError(const Eigen::Matrix4d &pose0,
                    const Eigen::Matrix4d &pose3) {
  const Eigen::Matrix4d measured = translationPose(3.0, 0.0, 0.0);
  const Eigen::Matrix4d predicted = pose0.inverse() * pose3;
  return (predicted.inverse() * measured).topRightCorner<3, 1>().norm();
}

} // namespace

int main() {
  std::cout << std::setprecision(6);

  PointCloud block01 = select({
      0, 1, 2, 16, 17, 18, 32, 33, 34, 48, 49, 50});
  PointCloud block12 = select({
      16, 17, 18, 20, 21, 22, 32, 33, 34, 36, 37, 38});
  PointCloud block23 = select({
      32, 33, 34, 36, 37, 38, 48, 49, 50, 52, 53, 54});
  PointCloud closure = block01;
  for (int point : {48, 49, 50, 52, 53, 54, 60, 61, 62}) {
    closure.push_back(gridPoints()[static_cast<std::size_t>(point)]);
  }

  const std::vector<Eigen::Matrix4d> truth{
      translationPose(0, 0, 0), translationPose(1, 0, 0),
      translationPose(2, 0, 0), translationPose(3, 0, 0)};
  const std::vector<Eigen::Matrix4d> initial{
      translationPose(0, 0, 0), translationPose(1.12, 0.04, -0.03),
      translationPose(2.24, 0.08, 0.02), translationPose(3.38, 0.12, 0.0)};
  const std::vector<PointCloud> globalScans{block01, block12, block23,
                                            closure};

  std::vector<PointCloud> scans(4);
  for (int i = 0; i < 4; ++i) {
    scans[i] = localFromGlobal(globalScans[i], truth[i]);
  }

  const std::vector<ScanEdgeSpec> edges{
      {0, 1, EdgeKind::Adjacent, 1.0, 1.0},
      {1, 2, EdgeKind::Adjacent, 1.0, 1.0},
      {2, 3, EdgeKind::Adjacent, 1.0, 1.0},
      {0, 3, EdgeKind::LoopClosure, 1.0, 1.0}};

  MultiScanConfig config;
  config.icp.maxIterations = 80;
  config.icp.maxCorrespondenceDistance = 0.5;
  config.icp.translationTolerance = 1e-10;
  config.icp.rotationTolerance = 1e-10;
  config.maxIterations = 80;
  config.convergenceTolerance = 1e-12;
  config.huberDelta = 0.1;

  std::cout << "=== 四站扫描闭环联合校正示例 ===\n";
  std::cout << "校正前各站位姿误差合计: " << poseError(initial, truth)
            << "\n";
  std::cout << "校正前闭环平移不闭合量: " << closureError(initial[0], initial[3])
            << "\n\n";

  const MultiScanResult result =
      correctMultiScan(scans, initial, edges, config);

  for (const EdgeStatus &status : result.edgeStatuses) {
    std::cout << toString(status.kind) << " 边 " << status.sourceStation
              << "->" << status.targetStation << ": "
              << (status.accepted ? "接受" : "拒绝")
              << ", ICP=" << status.reason << "\n";
  }
  std::cout << "\n终止原因: " << toString(result.reason)
            << (result.success ? "（成功）" : "（未成功）") << " - "
            << result.message << "\n";
  std::cout << "迭代次数: " << result.iterations << "\n";
  std::cout << "目标值: " << result.initialObjective << " -> "
            << result.finalObjective << "\n";
  std::cout << "校正后各站位姿误差合计: " << poseError(result.poses, truth)
            << "\n";
  std::cout << "校正后闭环平移不闭合量: "
            << closureError(result.poses[0], result.poses[3]) << "\n";
  std::cout << "融合点云点数: " << result.fusedCloud.size() << "\n";

  return result.success ? 0 : 1;
}
