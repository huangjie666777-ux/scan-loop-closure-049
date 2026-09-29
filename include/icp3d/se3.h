#pragma once

#include <Eigen/Dense>

#include "icp3d/rigid.h"

namespace icp3d {

// 六维扭量：前三维为旋转 w（旋转向量），后三维为平移 v（全局约定）。
using Twist = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

// SE(3) 逆变换。
RigidTransform inverse(const RigidTransform &transform);

// 变换复合 a * b（先作用 b 再作用 a）。
RigidTransform multiply(const RigidTransform &a, const RigidTransform &b);

// SE(3) 指数映射：T = exp(twist)，旋转由旋转向量经 Rodrigues 给出。
RigidTransform se3Exp(const Twist &twist);

// SE(3) 对数映射：twist = log(T)，逆运算，满足 log(exp(x)) 归一到最短弧。
Twist se3Log(const RigidTransform &transform);

// 左乘雅可比 J_l(w) 及其逆（w 为旋转向量部分）。
Matrix6d se3LeftJacobian(const Eigen::Vector3d &w);
Matrix6d se3LeftJacobianInverse(const Eigen::Vector3d &w);

// SE(3) 伴随矩阵：Ad_T = [[R, 0],[hat(t) R, R]]（扭量顺序 [w; v]）。
Matrix6d adjoint(const RigidTransform &transform);

} // namespace icp3d
