#include "icp3d/multiscan.h"

#include <cmath>
#include <queue>
#include <stdexcept>

#include "icp3d/icp.h"
#include "icp3d/point_cloud_map.h"
#include "icp3d/rigid.h"

namespace icp3d {

namespace {

void validateFiniteCloud(const PointCloud &cloud) {
  if (cloud.empty()) {
    throw std::invalid_argument("站点云不能为空");
  }
  for (const Point &point : cloud) {
    if (!point.allFinite()) {
      throw std::invalid_argument("站点云含非有限坐标");
    }
  }
}

bool connected(std::size_t stationCount,
               const std::vector<PoseGraphEdge> &edges) {
  std::vector<std::vector<int>> adjacency(stationCount);
  for (const PoseGraphEdge &edge : edges) {
    adjacency[edge.source].push_back(edge.target);
    adjacency[edge.target].push_back(edge.source);
  }
  std::vector<bool> visited(stationCount, false);
  std::queue<int> pending;
  pending.push(0);
  visited[0] = true;
  std::size_t reached = 1;
  while (!pending.empty()) {
    const int station = pending.front();
    pending.pop();
    for (int next : adjacency[station]) {
      if (!visited[next]) {
        visited[next] = true;
        ++reached;
        pending.push(next);
      }
    }
  }
  return reached == stationCount;
}

} // namespace

const char *toString(MultiScanTermination reason) {
  switch (reason) {
  case MultiScanTermination::Converged:
    return "Converged";
  case MultiScanTermination::MaxIterationsReached:
    return "MaxIterationsReached";
  case MultiScanTermination::NumericalFailure:
    return "NumericalFailure";
  case MultiScanTermination::GraphDisconnected:
    return "GraphDisconnected";
  }
  return "Unknown";
}

const char *toString(EdgeKind kind) {
  return kind == EdgeKind::LoopClosure ? "LoopClosure" : "Adjacent";
}

MultiScanResult correctMultiScan(
    const std::vector<PointCloud> &scans,
    const std::vector<Eigen::Matrix4d> &initialPoses,
    const std::vector<ScanEdgeSpec> &edgeSpecs,
    const MultiScanConfig &config) {
  if (scans.size() != initialPoses.size() || scans.size() < 2) {
    throw std::invalid_argument("点云数与初始位姿数必须相同且至少包含两站");
  }
  if (config.maxIterations <= 0 ||
      !std::isfinite(config.convergenceTolerance) ||
      config.convergenceTolerance <= 0.0 ||
      !std::isfinite(config.huberDelta) || config.huberDelta <= 0.0) {
    throw std::invalid_argument("非法多站校正参数");
  }
  for (const PointCloud &cloud : scans) {
    validateFiniteCloud(cloud);
  }

  std::vector<Se3Pose> initialSe3Poses;
  initialSe3Poses.reserve(initialPoses.size());
  for (const Eigen::Matrix4d &pose : initialPoses) {
    initialSe3Poses.push_back(poseFromMatrix4(pose));
  }

  for (const ScanEdgeSpec &edge : edgeSpecs) {
    if (edge.sourceStation < 0 || edge.targetStation < 0 ||
        edge.sourceStation >= static_cast<int>(scans.size()) ||
        edge.targetStation >= static_cast<int>(scans.size())) {
      throw std::invalid_argument("边包含无效站号");
    }
    if (edge.sourceStation == edge.targetStation) {
      throw std::invalid_argument("自环边不允许进入位姿图");
    }
    if (!(edge.translationWeight > 0.0) || !(edge.rotationWeight > 0.0) ||
        !std::isfinite(edge.translationWeight) ||
        !std::isfinite(edge.rotationWeight)) {
      throw std::invalid_argument("边权重必须为正有限值");
    }
  }

  MultiScanResult result;
  result.poses = initialPoses;
  std::vector<PoseGraphEdge> acceptedEdges;
  acceptedEdges.reserve(edgeSpecs.size());
  result.edgeStatuses.reserve(edgeSpecs.size());

  for (const ScanEdgeSpec &spec : edgeSpecs) {
    const Se3Pose sourcePose = initialSe3Poses[spec.sourceStation];
    const Se3Pose targetPose = initialSe3Poses[spec.targetStation];
    const Se3Pose initialRelative =
        multiply(inverse(targetPose), sourcePose);

    EdgeStatus status;
    status.sourceStation = spec.sourceStation;
    status.targetStation = spec.targetStation;
    status.kind = spec.kind;

    IcpResult icpResult = alignPointToPoint(
        scans[spec.sourceStation], scans[spec.targetStation],
        poseToMatrix4(initialRelative), config.icp);

    status.iterations = icpResult.iterations;
    status.numCorrespondences = icpResult.numCorrespondences;
    status.rms = icpResult.rms;
    status.measurement = icpResult.transform;
    status.reason = icpResult.converged
                         ? "Accepted: ICP converged"
                         : std::string("Rejected: ") +
                               toString(icpResult.reason) + " - " +
                               icpResult.message;
    status.accepted = icpResult.converged;

    if (icpResult.converged) {
      PoseGraphEdge edge;
      edge.source = spec.sourceStation;
      edge.target = spec.targetStation;
      edge.measurement = poseFromMatrix4(icpResult.transform);
      edge.translationWeight = spec.translationWeight;
      edge.rotationWeight = spec.rotationWeight;
      edge.loopClosure = spec.kind == EdgeKind::LoopClosure;
      acceptedEdges.push_back(edge);
    }
    result.edgeStatuses.push_back(status);
  }

  if (!connected(scans.size(), acceptedEdges)) {
    result.success = false;
    result.reason = MultiScanTermination::GraphDisconnected;
    result.message = "有效边图不连通，拒绝任取漂浮分量坐标";
    return result;
  }

  PoseGraphConfig graphConfig;
  graphConfig.maxIterations = config.maxIterations;
  graphConfig.convergenceTolerance = config.convergenceTolerance;
  graphConfig.huberDelta = config.huberDelta;
  PoseGraphResult graphResult =
      optimizePoseGraph(initialSe3Poses, acceptedEdges, graphConfig);

  result.initialObjective = graphResult.initialObjective;
  result.finalObjective = graphResult.finalObjective;
  result.iterations = graphResult.iterations;
  result.poses.clear();
  result.poses.reserve(graphResult.poses.size());
  for (const Se3Pose &pose : graphResult.poses) {
    result.poses.push_back(poseToMatrix4(pose));
  }
  result.fusedCloud = mergePointClouds(scans, graphResult.poses);

  switch (graphResult.reason) {
  case PoseGraphTermination::Converged:
    result.success = true;
    result.reason = MultiScanTermination::Converged;
    break;
  case PoseGraphTermination::MaxIterationsReached:
    result.success = false;
    result.reason = MultiScanTermination::MaxIterationsReached;
    break;
  case PoseGraphTermination::NumericalFailure:
    result.success = false;
    result.reason = MultiScanTermination::NumericalFailure;
    break;
  }
  result.message = graphResult.message;
  return result;
}

} // namespace icp3d
