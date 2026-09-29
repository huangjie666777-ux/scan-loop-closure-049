#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "icp3d/pose_graph_optimizer.h"
#include "icp3d/se3.h"
#include "icp3d/types.h"

namespace icp3d {

enum class EdgeKind {
  Adjacent,
  LoopClosure
};

struct ScanEdgeSpec {
  int sourceStation = 0;
  int targetStation = 0;
  EdgeKind kind = EdgeKind::Adjacent;
  double translationWeight = 1.0;
  double rotationWeight = 1.0;
};

struct MultiScanConfig {
  IcpConfig icp;
  int maxIterations = 50;
  double convergenceTolerance = 1e-10;
  double huberDelta = 1.0;
};

enum class MultiScanTermination {
  Converged,
  MaxIterationsReached,
  NumericalFailure,
  GraphDisconnected
};

struct EdgeStatus {
  int sourceStation = 0;
  int targetStation = 0;
  EdgeKind kind = EdgeKind::Adjacent;
  bool accepted = false;
  Eigen::Matrix4d measurement = Eigen::Matrix4d::Identity();
  int iterations = 0;
  std::size_t numCorrespondences = 0;
  double rms = 0.0;
  std::string reason;
};

struct MultiScanResult {
  bool success = false;
  std::vector<Eigen::Matrix4d> poses;
  PointCloud fusedCloud;
  std::vector<EdgeStatus> edgeStatuses;
  double initialObjective = 0.0;
  double finalObjective = 0.0;
  int iterations = 0;
  MultiScanTermination reason = MultiScanTermination::NumericalFailure;
  std::string message;
};

const char *toString(MultiScanTermination reason);
const char *toString(EdgeKind kind);

// 只使用调用者明确给出的重叠边；ICP 失败的边不进入位姿图，也不自动搜索。
MultiScanResult correctMultiScan(
    const std::vector<PointCloud> &scans,
    const std::vector<Eigen::Matrix4d> &initialPoses,
    const std::vector<ScanEdgeSpec> &edgeSpecs,
    const MultiScanConfig &config);

} // namespace icp3d
