#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "icp3d/icp.h"
#include "icp3d/kdtree.h"
#include "icp3d/multiscan.h"
#include "icp3d/rigid.h"
#include "icp3d/se3.h"

using namespace icp3d;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string &name) {
  ++g_checks;
  if (condition) {
    std::cout << "  [PASS] " << name << "\n";
  } else {
    ++g_failures;
    std::cout << "  [FAIL] " << name << "\n";
  }
}

void checkThrows(const std::function<void()> &fn, const std::string &name) {
  bool thrown = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    thrown = true;
  } catch (const std::exception &) {
  }
  check(thrown, name);
}

Eigen::Matrix4d makeTransform(const Eigen::Matrix3d &rotation,
                              const Eigen::Vector3d &translation) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.topLeftCorner<3, 3>() = rotation;
  transform.topRightCorner<3, 1>() = translation;
  return transform;
}

Eigen::Matrix4d translationPose(double x, double y, double z) {
  return makeTransform(Eigen::Matrix3d::Identity(),
                       Eigen::Vector3d(x, y, z));
}

PointCloud makeGlobalGrid() {
  PointCloud points;
  for (int x = 0; x <= 4; ++x) {
    for (int y = 0; y <= 4; ++y) {
      for (int z = 0; z <= 4; ++z) {
        points.emplace_back(static_cast<double>(x), static_cast<double>(y),
                            static_cast<double>(z));
      }
    }
  }
  return points;
}

PointCloud selectGlobalPoints(const std::vector<int> &indices) {
  const PointCloud global = makeGlobalGrid();
  PointCloud selected;
  for (int index : indices) {
    selected.push_back(global[static_cast<std::size_t>(index)]);
  }
  return selected;
}

PointCloud toLocal(const PointCloud &global, const Eigen::Matrix4d &pose) {
  PointCloud local;
  const Eigen::Matrix4d inversePose = pose.inverse();
  for (const Point &point : global) {
    Eigen::Vector4d homogeneous(point.x(), point.y(), point.z(), 1.0);
    local.push_back((inversePose * homogeneous).head<3>());
  }
  return local;
}

void testKdTree() {
  std::cout << "[KdTree]\n";
  PointCloud points;
  points.emplace_back(0, 0, 0);   // 0
  points.emplace_back(2, 0, 0);   // 1
  points.emplace_back(-2, 0, 0);  // 2
  points.emplace_back(0, 3, 0);   // 3
  points.emplace_back(0, -3, 0);  // 4
  KdTree tree(points);
  const auto nearest = tree.nearest(Point(0.9, 0.0, 0.0));
  check(nearest.index == 0 && std::abs(nearest.squaredDistance - 0.81) < 1e-12,
        "最近点查询返回正确索引与平方距离");

  // 查询点 (1,0,0) 与索引 0、1 等距，应返回较小原始索引。
  const auto tie = tree.nearest(Point(1.0, 0.0, 0.0));
  check(tie.index == 0, "等距时返回目标原始索引较小者");
}

void testRigidEstimate() {
  std::cout << "[RigidEstimate]\n";
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized())
          .toRotationMatrix();
  const Eigen::Vector3d translation(1.5, -2.0, 0.4);

  PointCloud source{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}, {2, -1, 0.5}};
  std::vector<Correspondence> pairs;
  for (const auto &point : source) {
    pairs.push_back({point, rotation * point + translation});
  }
  const RigidTransform estimated = estimateRigid(pairs);
  check(estimated.rotation.isApprox(rotation, 1e-10), "SVD 恢复已知旋转");
  check(estimated.translation.isApprox(translation, 1e-10), "SVD 恢复已知平移");
  check(std::abs(estimated.rotation.determinant() - 1.0) < 1e-12,
        "估计旋转行列式为 1");
  check((estimated.rotation.transpose() * estimated.rotation -
         Eigen::Matrix3d::Identity())
            .norm() < 1e-12,
        "估计旋转保持正交");

  bool failedForFew = false;
  try {
    estimateRigid({pairs[0], pairs[1]});
  } catch (const std::runtime_error &) {
    failedForFew = true;
  }
  check(failedForFew, "配对不足 3 组时明确失败");

  // 共线点：旋转不可唯一确定，必须判为退化。
  std::vector<Correspondence> collinear;
  for (double t : {0.0, 1.0, 2.0, 3.0}) {
    Eigen::Vector3d point(t, 0.0, 0.0);
    collinear.push_back({point, rotation * point + translation});
  }
  bool failedForDegenerate = false;
  try {
    estimateRigid(collinear);
  } catch (const std::runtime_error &) {
    failedForDegenerate = true;
  }
  check(failedForDegenerate, "共线几何退化时报错而非任取变换");

  // compose: T_new = Δ * T_old。
  RigidTransform pose{RigidTransform{rotation, translation}};
  RigidTransform delta{RigidTransform{
      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()).toRotationMatrix(),
      Eigen::Vector3d(0.1, 0.2, 0.3)}};
  const RigidTransform composed = compose(delta, pose);
  const auto m = toMatrix4(composed);
  const Eigen::Matrix4d expected = toMatrix4(delta) * toMatrix4(pose);
  check(m.isApprox(expected), "增量按 Δ*T 正确累加");
  check(toMatrix4(fromMatrix4(m)).isApprox(m), "齐次矩阵互逆转换一致");

  bool invalidRotation = false;
  try {
    Eigen::Matrix3d mirror = Eigen::Matrix3d::Identity();
    mirror(2, 2) = -1.0;
    validateRotation(mirror);
  } catch (const std::invalid_argument &) {
    invalidRotation = true;
  }
  check(invalidRotation, "镜像旋转被拒绝");

  bool scaledRotation = false;
  try {
    Eigen::Matrix3d scaled = rotation * 2.0;
    validateRotation(scaled);
  } catch (const std::invalid_argument &) {
    scaledRotation = true;
  }
  check(scaledRotation, "含缩放的旋转被拒绝");

  PointCloud planarSource{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  std::vector<Correspondence> planarPairs;
  for (const Point &point : planarSource) {
    planarPairs.push_back({point, rotation * point + translation});
  }
  const RigidTransform planarEstimate = estimateRigid(planarPairs);
  check(planarEstimate.rotation.isApprox(rotation, 1e-10) &&
            planarEstimate.translation.isApprox(translation, 1e-10),
        "非共线平面三点不被误判退化");
}

void testSe3() {
  std::cout << "[SE3]\n";
  Se3Tangent tangent;
  tangent << 0.4, -0.2, 0.3, 0.08, -0.05, 0.12;
  const Se3Pose pose = expSe3(tangent);
  check(logSe3(pose).isApprox(tangent, 1e-10), "SE3 exp/log 往返一致");

  const Se3Pose a{Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitY())
                      .toRotationMatrix(),
                  Eigen::Vector3d(1.0, 0.0, -0.5)};
  const Se3Pose b{Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitX())
                      .toRotationMatrix(),
                  Eigen::Vector3d(-0.5, 0.4, 0.8)};
  const Se3Pose ab = multiply(a, b);
  const Se3Pose identity = multiply(inverse(a), a);
  check(poseToMatrix4(ab).isApprox(poseToMatrix4(a) * poseToMatrix4(b)),
        "SE3 组合与齐次矩阵乘法一致");
  check(poseToMatrix4(identity).isApprox(Eigen::Matrix4d::Identity(), 1e-12),
        "SE3 逆变换正确");
  check(relativeLog(a, a, identity).norm() < 1e-12,
        "测量等于预测时 SE3 对数残差为零");
}

void testPoseGraph() {
  std::cout << "[PoseGraph]\n";
  std::vector<Se3Pose> truePoses(4);
  std::vector<Se3Pose> driftedPoses(4);
  for (int i = 0; i < 4; ++i) {
    truePoses[i].translation =
        Eigen::Vector3d(static_cast<double>(i), 0.0, 0.0);
    driftedPoses[i].translation =
        Eigen::Vector3d(1.05 * static_cast<double>(i), 0.0, 0.0);
  }

  std::vector<PoseGraphEdge> edges;
  auto addEdge = [&](int source, int target, bool loop) {
    PoseGraphEdge edge;
    edge.source = source;
    edge.target = target;
    edge.measurement =
        multiply(inverse(truePoses[target]), truePoses[source]);
    edge.loopClosure = loop;
    edges.push_back(edge);
  };
  addEdge(0, 1, false);
  addEdge(1, 2, false);
  addEdge(2, 3, false);
  addEdge(0, 3, true);

  PoseGraphConfig config;
  config.maxIterations = 50;
  config.convergenceTolerance = 1e-12;
  config.huberDelta = 1.0;
  const PoseGraphResult result =
      optimizePoseGraph(driftedPoses, edges, config);
  check(result.converged && result.finalObjective < result.initialObjective,
        "累计漂移经闭环联合优化后目标下降");
  bool recovered = true;
  for (int i = 0; i < 4; ++i) {
    recovered &= result.poses[i].translation.isApprox(
        truePoses[i].translation, 1e-6);
  }
  check(recovered, "联合优化恢复各站平移而非逐边串乘");
  check(poseToMatrix4(result.poses[0]).isApprox(Eigen::Matrix4d::Identity()),
        "首站位姿在优化中固定");
}

PointCloud makeSphereCloud(std::size_t count) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  PointCloud cloud;
  while (cloud.size() < count) {
    Point point(uniform(rng), uniform(rng), uniform(rng));
    if (point.norm() <= 1.0) {
      cloud.push_back(point * 1.5);
    }
  }
  return cloud;
}

void testIcp() {
  std::cout << "[ICP]\n";
  const PointCloud target = makeSphereCloud(300);
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(0.25, Eigen::Vector3d(1.0, -1.0, 1.0).normalized())
          .toRotationMatrix();
  const Eigen::Vector3d translation(0.3, -0.2, 0.15);
  const Eigen::Matrix4d truth = makeTransform(rotation, translation);
  const Eigen::Matrix4d truthInv = truth.inverse();

  PointCloud source;
  for (std::size_t i = 0; i < target.size(); i += 4) {
    Eigen::Vector4d homogeneous(target[i].x(), target[i].y(),
                                target[i].z(), 1.0);
    source.push_back((truthInv * homogeneous).head<3>());
  }
  const PointCloud sourceCopy = source;
  const PointCloud targetCopy = target;

  IcpConfig config;
  config.maxIterations = 100;
  config.maxCorrespondenceDistance = 0.5;
  config.translationTolerance = 1e-9;
  config.rotationTolerance = 1e-9;

  const Eigen::Matrix4d initial = Eigen::Matrix4d::Identity();
  const IcpResult result = alignPointToPoint(source, target, initial, config);
  check(result.converged, "已知变换场景下 ICP 收敛");
  check(result.reason == TerminationReason::Converged, "终止原因标记为 Converged");
  check(result.transform.isApprox(truth, 1e-6), "最终变换恢复已知刚体变换");
  check(result.rms < 1e-5, "配准后 RMS 接近零");
  check(result.numCorrespondences == source.size(),
        "所有正常源点形成有效配对");
  check(source == sourceCopy && target == targetCopy, "配准过程不修改输入点云");

  // 离群点：应被最大对应距离剔除，配准仍成功。
  PointCloud sourceWithOutliers = source;
  sourceWithOutliers.emplace_back(20.0, 20.0, 20.0);
  sourceWithOutliers.emplace_back(-20.0, -20.0, -20.0);
  const IcpResult resultOutliers =
      alignPointToPoint(sourceWithOutliers, target, initial, config);
  check(resultOutliers.converged &&
            resultOutliers.numCorrespondences == source.size(),
        "离群源点被剔除且不影响配准");

  // 阈值过小：无有效配对，明确失败而非冒充成功。
  IcpConfig noPairs = config;
  noPairs.maxCorrespondenceDistance = 1e-9;
  const IcpResult resultNoPairs =
      alignPointToPoint(source, target, initial, noPairs);
  check(!resultNoPairs.converged &&
            resultNoPairs.reason ==
                TerminationReason::InsufficientCorrespondences,
        "有效配对不足时返回 InsufficientCorrespondences");

  // 达到次数上限：未收敛，返回当前估计与重新匹配的指标。
  IcpConfig capped = config;
  capped.maxIterations = 1;
  capped.translationTolerance = 1e-15;
  capped.rotationTolerance = 1e-15;
  const IcpResult resultCapped =
      alignPointToPoint(source, target, initial, capped);
  check(!resultCapped.converged && resultCapped.iterations == 1 &&
            resultCapped.reason == TerminationReason::MaxIterationsReached,
        "达到次数上限返回未收敛及当前估计");
}

void testMultiScan() {
  std::cout << "[MultiScan]\n";
  std::vector<PointCloud> globals(4);
  std::vector<int> shared;
  for (int y = 0; y <= 2; ++y) {
    for (int z = 0; z <= 2; ++z) {
      shared.push_back(y * 25 + z * 5);
    }
  }
  globals[0] = selectGlobalPoints(shared);
  globals[1] = selectGlobalPoints({
      25, 26, 27, 30, 31, 32, 50, 51, 52, 55, 56, 57});
  globals[2] = selectGlobalPoints({
      50, 51, 52, 55, 56, 57, 75, 76, 77, 80, 81, 82});
  globals[3] = selectGlobalPoints(shared);
  for (int &index : shared) {
    index += 75;
  }
  for (int point : shared) {
    globals[3].push_back(makeGlobalGrid()[static_cast<std::size_t>(point)]);
  }

  std::vector<Eigen::Matrix4d> truePoses{
      translationPose(0, 0, 0), translationPose(1, 0, 0),
      translationPose(2, 0, 0), translationPose(3, 0, 0)};
  std::vector<Eigen::Matrix4d> initialPoses{
      translationPose(0, 0, 0), translationPose(1.08, 0.02, -0.02),
      translationPose(2.18, 0.03, 0.01), translationPose(3.30, 0.0, 0.0)};

  std::vector<PointCloud> scans(4);
  for (int i = 0; i < 4; ++i) {
    scans[i] = toLocal(globals[i], truePoses[i]);
  }
  const auto initialCopy = initialPoses;

  std::vector<ScanEdgeSpec> specs{
      {0, 1, EdgeKind::Adjacent, 1.0, 1.0},
      {1, 2, EdgeKind::Adjacent, 1.0, 1.0},
      {2, 3, EdgeKind::Adjacent, 1.0, 1.0},
      {0, 3, EdgeKind::LoopClosure, 1.0, 1.0}};

  MultiScanConfig config;
  config.icp.maxIterations = 80;
  config.icp.maxCorrespondenceDistance = 0.4;
  config.icp.translationTolerance = 1e-9;
  config.icp.rotationTolerance = 1e-9;
  config.maxIterations = 80;
  config.convergenceTolerance = 1e-12;
  config.huberDelta = 1.0;

  const MultiScanResult result =
      correctMultiScan(scans, initialPoses, specs, config);
  check(result.success && result.reason == MultiScanTermination::Converged,
        "含闭环的多站校正收敛");
  check(result.edgeStatuses.size() == 4 &&
            result.edgeStatuses[3].kind == EdgeKind::LoopClosure &&
            result.edgeStatuses[3].accepted,
        "逐边原因和闭环类型被返回");
  bool poseRecovered = true;
  double fusedError = 0.0;
  for (int i = 0; i < 4; ++i) {
    poseRecovered &= result.poses[i].isApprox(truePoses[i], 1e-6);
  }
  std::size_t fusedPointIndex = 0;
  for (std::size_t station = 0; station < scans.size(); ++station) {
    for (const Point &localPoint : scans[station]) {
      Eigen::Vector4d homogeneous(localPoint.x(), localPoint.y(),
                                  localPoint.z(), 1.0);
      const Point fused = (result.poses[station] * homogeneous).head<3>();
      fusedError += (fused - result.fusedCloud[fusedPointIndex++]).norm();
    }
  }
  check(poseRecovered, "闭环校正恢复各站全局位姿");
  check(fusedError == 0.0 &&
            result.fusedCloud.size() == scans[0].size() + scans[1].size() +
                                    scans[2].size() + scans[3].size(),
        "按最终联合位姿拼接全部原始点");
  check(result.finalObjective < result.initialObjective,
        "返回初末目标值且末值下降");
  bool inputUnchanged = (initialPoses == initialCopy);
  check(inputUnchanged, "重复使用的初始位姿输入不被修改");

  const std::vector<ScanEdgeSpec> adjacentOnly{
      {0, 1, EdgeKind::Adjacent, 1.0, 1.0},
      {2, 3, EdgeKind::Adjacent, 1.0, 1.0}};
  const MultiScanResult missingLoop =
      correctMultiScan(scans, initialPoses, adjacentOnly, config);
  check(!missingLoop.success &&
            missingLoop.reason == MultiScanTermination::GraphDisconnected,
        "有效边图不连通时明确失败");

  const MultiScanResult repeated =
      correctMultiScan(scans, initialPoses, specs, config);
  check(repeated.success && repeated.initialObjective == result.initialObjective,
        "重复调用互不污染");

  bool rejectsSelfLoop = false;
  try {
    auto invalid = specs;
    invalid[0].targetStation = 0;
    correctMultiScan(scans, initialPoses, invalid, config);
  } catch (const std::invalid_argument &) {
    rejectsSelfLoop = true;
  }
  check(rejectsSelfLoop, "拒绝自环边");
}

void testValidation() {
  std::cout << "[Validation]\n";
  PointCloud cloud;
  cloud.emplace_back(0, 0, 0);
  cloud.emplace_back(1, 1, 1);
  PointCloud bad = cloud;
  bad[0].x() = std::numeric_limits<double>::infinity();
  const auto identity = Eigen::Matrix4d::Identity();
  IcpConfig config;

  PointCloud empty;
  checkThrows([&] { alignPointToPoint(empty, cloud, identity, config); },
              "拒绝空源点云");
  checkThrows([&] { alignPointToPoint(cloud, empty, identity, config); },
              "拒绝空目标点云");
  checkThrows([&] { alignPointToPoint(bad, cloud, identity, config); },
              "拒绝非有限坐标");

  Eigen::Matrix4d mirror = Eigen::Matrix4d::Identity();
  mirror(2, 2) = -1.0;
  checkThrows([&] { alignPointToPoint(cloud, cloud, mirror, config); },
              "拒绝镜像初始变换");

  IcpConfig badIterations = config;
  badIterations.maxIterations = 0;
  checkThrows(
      [&] { alignPointToPoint(cloud, cloud, identity, badIterations); },
      "拒绝非正最大迭代次数");
  IcpConfig badDistance = config;
  badDistance.maxCorrespondenceDistance = -1.0;
  checkThrows(
      [&] { alignPointToPoint(cloud, cloud, identity, badDistance); },
      "拒绝非正最大对应距离");
  IcpConfig badTolerance = config;
  badTolerance.translationTolerance = 0.0;
  checkThrows(
      [&] { alignPointToPoint(cloud, cloud, identity, badTolerance); },
      "拒绝非正平移阈值");
}

} // namespace

int main() {
  testKdTree();
  testRigidEstimate();
  testSe3();
  testPoseGraph();
  testIcp();
  testMultiScan();
  testValidation();
  std::cout << "\n共 " << g_checks << " 项检查，失败 " << g_failures
            << " 项\n";
  return g_failures == 0 ? 0 : 1;
}
