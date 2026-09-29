#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "icp3d/se3.h"

namespace icp3d {

enum class PoseGraphTermination {
  Converged,
  MaxIterationsReached,
  NumericalFailure
};

struct PoseGraphEdge {
  int source = 0;
  int target = 0;
  Se3Pose measurement;
  double translationWeight = 1.0;
  double rotationWeight = 1.0;
  bool loopClosure = false;
};

struct PoseGraphConfig {
  int maxIterations = 50;
  double convergenceTolerance = 1e-10;
  double huberDelta = 1.0;
};

struct PoseGraphResult {
  bool converged = false;
  std::vector<Se3Pose> poses;
  double initialObjective = 0.0;
  double finalObjective = 0.0;
  int iterations = 0;
  PoseGraphTermination reason = PoseGraphTermination::NumericalFailure;
  std::string message;
};

const char *toString(PoseGraphTermination reason);

// poses[0] 固定，其余站点联合优化。调用前需确认边构成的无向图连通。
PoseGraphResult optimizePoseGraph(std::vector<Se3Pose> poses,
                                  const std::vector<PoseGraphEdge> &edges,
                                  const PoseGraphConfig &config);

} // namespace icp3d
