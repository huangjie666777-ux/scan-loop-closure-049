#include "icp3d/point_cloud_map.h"

#include <stdexcept>

namespace icp3d {

PointCloud transformPointCloud(const PointCloud &cloud, const Se3Pose &pose) {
  PointCloud transformed;
  transformed.reserve(cloud.size());
  for (const Point &point : cloud) {
    transformed.push_back(pose.rotation * point + pose.translation);
  }
  return transformed;
}

PointCloud mergePointClouds(const std::vector<PointCloud> &clouds,
                            const std::vector<Se3Pose> &poses) {
  PointCloud fused;
  if (clouds.size() != poses.size()) {
    throw std::invalid_argument("点云数量与位姿数量不一致");
  }
  std::size_t totalPoints = 0;
  for (const PointCloud &cloud : clouds) {
    totalPoints += cloud.size();
  }
  fused.reserve(totalPoints);
  for (std::size_t station = 0; station < clouds.size(); ++station) {
    PointCloud transformed = transformPointCloud(clouds[station],
                                                 poses[station]);
    fused.insert(fused.end(), transformed.begin(), transformed.end());
  }
  return fused;
}

} // namespace icp3d
