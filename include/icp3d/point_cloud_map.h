#pragma once

#include <Eigen/Dense>

#include <vector>

#include "icp3d/se3.h"
#include "icp3d/types.h"

namespace icp3d {

PointCloud transformPointCloud(const PointCloud &cloud, const Se3Pose &pose);
PointCloud mergePointClouds(const std::vector<PointCloud> &clouds,
                            const std::vector<Se3Pose> &poses);

} // namespace icp3d
