#include "icp3d/pose_graph_optimizer.h"

#include <cmath>
#include <limits>

#include "icp3d/rigid.h"

namespace icp3d {

namespace {

using ResidualVector = Eigen::VectorXd;

bool validPoses(const std::vector<Se3Pose> &poses) {
  for (const Se3Pose &pose : poses) {
    if (!pose.rotation.allFinite() || !pose.translation.allFinite()) {
      return false;
    }
  }
  return true;
}

Eigen::Vector<double, 6> weightedResidual(const PoseGraphEdge &edge,
                                          const std::vector<Se3Pose> &poses) {
  const Se3Tangent residual =
      relativeLog(edge.measurement, poses[edge.source], poses[edge.target]);
  Eigen::Vector<double, 6> weighted;
  weighted.head<3>() = edge.translationWeight * residual.head<3>();
  weighted.tail<3>() = edge.rotationWeight * residual.tail<3>();
  return weighted;
}

double huberCost(double norm, double delta) {
  if (norm <= delta) {
    return 0.5 * norm * norm;
  }
  return delta * (norm - 0.5 * delta);
}

double objective(const std::vector<PoseGraphEdge> &edges,
                 const std::vector<Se3Pose> &poses, double huberDelta) {
  double total = 0.0;
  for (const PoseGraphEdge &edge : edges) {
    const double norm = weightedResidual(edge, poses).norm();
    const double scale = edge.loopClosure ? huberDelta : 1.0;
    total += huberCost(norm, scale);
  }
  return total;
}

ResidualVector allResiduals(const std::vector<PoseGraphEdge> &edges,
                            const std::vector<Se3Pose> &poses) {
  ResidualVector residual(6 * static_cast<Eigen::Index>(edges.size()));
  for (std::size_t i = 0; i < edges.size(); ++i) {
    residual.segment<6>(6 * static_cast<Eigen::Index>(i)) =
        weightedResidual(edges[i], poses);
  }
  return residual;
}

void stationPerturbation(std::vector<Se3Pose> &poses, int station,
                         int component, double epsilon) {
  Se3Tangent delta = Se3Tangent::Zero();
  delta(component) = epsilon;
  poses[station] = multiply(expSe3(delta), poses[station]);
}

void buildJacobian(const std::vector<PoseGraphEdge> &edges,
                   const std::vector<Se3Pose> &poses, Eigen::MatrixXd &jacobian) {
  const std::size_t variableStations = poses.size() - 1;
  const double epsilon = 1e-6;
  for (std::size_t station = 1; station < poses.size(); ++station) {
    for (int component = 0; component < 6; ++component) {
      const std::size_t column = (station - 1) * 6 + component;
      std::vector<Se3Pose> plus = poses;
      std::vector<Se3Pose> minus = poses;
      const double finiteStep =
          epsilon * std::max(1.0, poses[station].translation.cwiseAbs().sum() /
                                      3.0);
      stationPerturbation(plus, static_cast<int>(station), component,
                          finiteStep);
      stationPerturbation(minus, static_cast<int>(station), component,
                          -finiteStep);
      const ResidualVector residualPlus = allResiduals(edges, plus);
      const ResidualVector residualMinus = allResiduals(edges, minus);
      jacobian.col(static_cast<Eigen::Index>(column)) =
          (residualPlus - residualMinus) / (2.0 * finiteStep);
    }
  }
  (void)variableStations;
}

std::vector<Se3Pose> applyStep(const std::vector<Se3Pose> &poses,
                               const Eigen::VectorXd &step) {
  std::vector<Se3Pose> candidate = poses;
  for (std::size_t station = 1; station < poses.size(); ++station) {
    const Eigen::Index begin = 6 * static_cast<Eigen::Index>(station - 1);
    const Se3Tangent delta = step.segment<6>(begin);
    candidate[station] = multiply(expSe3(delta), candidate[station]);
  }
  return candidate;
}

} // namespace

const char *toString(PoseGraphTermination reason) {
  switch (reason) {
  case PoseGraphTermination::Converged:
    return "Converged";
  case PoseGraphTermination::MaxIterationsReached:
    return "MaxIterationsReached";
  case PoseGraphTermination::NumericalFailure:
    return "NumericalFailure";
  }
  return "Unknown";
}

PoseGraphResult optimizePoseGraph(std::vector<Se3Pose> poses,
                                  const std::vector<PoseGraphEdge> &edges,
                                  const PoseGraphConfig &config) {
  PoseGraphResult result;
  if (config.maxIterations <= 0 || !std::isfinite(config.convergenceTolerance) ||
      config.convergenceTolerance <= 0.0 ||
      !std::isfinite(config.huberDelta) || config.huberDelta <= 0.0) {
    throw std::invalid_argument("非法位姿图优化参数");
  }
  if (poses.empty()) {
    throw std::invalid_argument("位姿图至少需要一个站点");
  }
  for (const Se3Pose &pose : poses) {
    validateRotation(pose.rotation);
    if (!pose.translation.allFinite()) {
      throw std::invalid_argument("站点平移含非有限值");
    }
  }
  for (const PoseGraphEdge &edge : edges) {
    if (edge.source < 0 || edge.target < 0 ||
        edge.source >= static_cast<int>(poses.size()) ||
        edge.target >= static_cast<int>(poses.size()) ||
        edge.source == edge.target) {
      throw std::invalid_argument("位姿图边包含非法站号或自环");
    }
    if (!(edge.translationWeight > 0.0) || !(edge.rotationWeight > 0.0) ||
        !std::isfinite(edge.translationWeight) ||
        !std::isfinite(edge.rotationWeight)) {
      throw std::invalid_argument("位姿图边权重必须为正有限值");
    }
    validateRotation(edge.measurement.rotation);
    if (!edge.measurement.translation.allFinite()) {
      throw std::invalid_argument("位姿图测量含非有限平移");
    }
  }

  result.poses = poses;
  result.initialObjective = objective(edges, poses, config.huberDelta);
  result.finalObjective = result.initialObjective;
  double currentObjective = result.initialObjective;

  const Eigen::Index variables = 6 * static_cast<Eigen::Index>(poses.size() - 1);
  if (variables == 0 || edges.empty() || currentObjective == 0.0) {
    result.converged = true;
    result.reason = PoseGraphTermination::Converged;
    result.message = "目标函数已为零或没有需要优化的变量";
    return result;
  }

  for (int iteration = 1; iteration <= config.maxIterations; ++iteration) {
    const ResidualVector residual = allResiduals(edges, poses);
    Eigen::MatrixXd jacobian(residual.size(), variables);
    buildJacobian(edges, poses, jacobian);

    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(variables, variables);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(variables);
    for (std::size_t i = 0; i < edges.size(); ++i) {
      const auto edgeResidual =
          residual.segment<6>(6 * static_cast<Eigen::Index>(i));
      const auto edgeJacobian =
          jacobian.block(6 * static_cast<Eigen::Index>(i), 0, 6, variables);
      const double norm = edgeResidual.norm();
      double robustWeight = 1.0;
      if (edges[i].loopClosure && norm > config.huberDelta) {
        robustWeight = config.huberDelta / norm;
      }
      hessian.noalias() += robustWeight * edgeJacobian.transpose() * edgeJacobian;
      gradient.noalias() += robustWeight * edgeJacobian.transpose() * edgeResidual;
    }

    const double diagonalScale = std::max(1.0, hessian.diagonal().maxCoeff());
    double damping = 1e-3 * diagonalScale;
    Eigen::VectorXd step;
    bool solved = false;
    for (int attempt = 0; attempt < 30; ++attempt) {
      Eigen::LLT<Eigen::MatrixXd> solver(
          hessian + damping * Eigen::MatrixXd::Identity(variables, variables));
      if (solver.info() == Eigen::Success) {
        step = solver.solve(-gradient);
        if (step.allFinite()) {
          solved = true;
          break;
        }
      }
      damping *= 4.0;
    }
    if (!solved) {
      result.poses = poses;
      result.iterations = iteration - 1;
      result.finalObjective = currentObjective;
      result.reason = PoseGraphTermination::NumericalFailure;
      result.message = "位姿图线性求解失败，保留最后有效估计";
      return result;
    }

    double stepScale = 1.0;
    std::vector<Se3Pose> candidate;
    double candidateObjective = std::numeric_limits<double>::infinity();
    bool accepted = false;
    for (int backtrack = 0; backtrack < 20; ++backtrack) {
      candidate = applyStep(poses, stepScale * step);
      if (validPoses(candidate)) {
        candidateObjective = objective(edges, candidate, config.huberDelta);
        if (std::isfinite(candidateObjective) &&
            candidateObjective <= currentObjective + 1e-12 *
                                    std::max(1.0, currentObjective)) {
          accepted = true;
          break;
        }
      }
      stepScale *= 0.5;
    }

    if (!accepted) {
      result.poses = poses;
      result.iterations = iteration - 1;
      result.finalObjective = currentObjective;
      result.reason = PoseGraphTermination::NumericalFailure;
      result.message = "找不到不增加目标函数的有效更新，保留最后有效估计";
      return result;
    }

    const double decrease = currentObjective - candidateObjective;
    poses = candidate;
    currentObjective = candidateObjective;
    result.iterations = iteration;

    if (std::isfinite(config.convergenceTolerance) &&
        decrease <= config.convergenceTolerance *
                       (1.0 + std::abs(result.initialObjective))) {
      result.poses = poses;
      result.finalObjective = currentObjective;
      result.converged = true;
      result.reason = PoseGraphTermination::Converged;
      result.message = "目标函数下降量低于收敛容差";
      return result;
    }
  }

  result.poses = poses;
  result.finalObjective = currentObjective;
  result.converged = false;
  result.reason = PoseGraphTermination::MaxIterationsReached;
  result.message = "达到最大迭代次数，返回当前估计";
  return result;
}

} // namespace icp3d
