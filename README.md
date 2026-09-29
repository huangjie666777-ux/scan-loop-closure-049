# 工件扫描点云刚体配准库 `icp3d`

可复用的 C++17 三维点云**点到点 ICP（Iterative Closest Point）**刚体配准库，
用于将两次或多次工件扫描对齐到同一坐标系。基于 Eigen 3.4.0，仅含头文件依赖，
构建为静态库 `libicp3d.a`。

## 功能范围

- 输入源点云、目标点云与源到目标的初始刚体变换；双精度、同单位坐标。
- 两点集大小可不同，无需预设对应关系；输入不会被修改。
- 只求解刚体变换（旋转 + 平移），**不求解缩放、不接受镜像**。
- ICP 是局部方法：**不承诺从任意初始位姿收敛到全局最优**，需提供合理初值。
- 支持多站闭环校正：只使用调用者明确指定的相邻边和闭环边，不自动搜索重叠。
- 每条边先调用现有 ICP 生成相对测量，失败或未收敛的边不入图并返回原因。
- 固定首站，联合优化其余站点 SE3 位姿；闭环边使用可配置 Huber 损失。
- 优化后按最终位姿拼接全部原始点，不将逐边配准结果简单串乘。

## 目录结构

```
include/icp3d/
  types.h    公共类型：Point/PointCloud、IcpConfig、IcpResult、终止原因
  kdtree.h   目标点云 3D KD 树（空间检索职责）
  rigid.h    刚体变换校验、组合与最小二乘估计（刚体估计职责）
  icp.h      点到点 ICP 主入口（迭代控制职责）
  se3.h      SE3 指数/对数、逆变换与相对对数残差
  pose_graph_optimizer.h  固定首站的 SE3 位姿图联合优化
  multiscan.h            多站配准建图、边过滤、连通性检查与结果
  point_cloud_map.h      点云刚体变换与最终拼接
src/
  kdtree.cpp / rigid.cpp / types.cpp / icp.cpp / se3.cpp
  pose_graph_optimizer.cpp / point_cloud_map.cpp / multiscan.cpp
examples/example.cpp            两站 ICP + 离群点示例
examples/multiscan_example.cpp  四站累计漂移 + 闭环约束示例
tests/test_icp.cpp              单元/集成测试（45 项检查，无第三方测试框架）
Makefile
```

## 构建与运行

环境：GCC 11.4、GNU Make，Eigen 3.4.0 位于 `third_party/eigen3`。

```bash
make            # 构建静态库、两个示例和 build/icp_tests
make test       # 运行全部测试
make example    # 运行两站与多站闭环示例
make clean
```

在其他项目中使用：编译时加 `-Iinclude -Ithird_party/eigen3`，链接 `-licp3d`。

## 公共接口

```cpp
#include "icp3d/icp.h"

icp3d::PointCloud source, target;                 // std::vector<Eigen::Vector3d>
Eigen::Matrix4d initial = Eigen::Matrix4d::Identity();

icp3d::IcpConfig config;
config.maxIterations             = 50;    // 最大迭代轮数（> 0）
config.maxCorrespondenceDistance = 1.0;   // 最大对应距离（与坐标同单位，> 0）
config.translationTolerance      = 1e-6;  // 平移增量收敛阈值（> 0）
config.rotationTolerance         = 1e-6;  // 旋转增量收敛阈值，弧度（> 0）
config.rankEpsilon               = 1e-10; // 旋转可解的奇异值判据（> 0）

icp3d::IcpResult result =
    icp3d::alignPointToPoint(source, target, initial, config);

// result.transform          最终（或失败时当前）源->目标 4x4 变换
// result.iterations         实际迭代轮数
// result.numCorrespondences 最终有效配对数
// result.rms                最终变换重新匹配的均方根距离
// result.converged / result.reason / result.message
```

## 多站闭环校正

```cpp
#include "icp3d/multiscan.h"

std::vector<icp3d::PointCloud> scans;
std::vector<Eigen::Matrix4d> initialPoses;
std::vector<icp3d::ScanEdgeSpec> edges;

edges.push_back({0, 1, icp3d::EdgeKind::Adjacent, 1.0, 1.0});
edges.push_back({1, 2, icp3d::EdgeKind::Adjacent, 1.0, 1.0});
edges.push_back({2, 3, icp3d::EdgeKind::Adjacent, 1.0, 1.0});
edges.push_back({0, 3, icp3d::EdgeKind::LoopClosure, 1.0, 1.0});

icp3d::MultiScanConfig config;
config.icp.maxCorrespondenceDistance = 0.5;
config.maxIterations = 80;
config.convergenceTolerance = 1e-12;
config.huberDelta = 0.1;

icp3d::MultiScanResult result =
    icp3d::correctMultiScan(scans, initialPoses, edges, config);

// result.poses / fusedCloud / edgeStatuses / initialObjective
// result.finalObjective / iterations / reason / message
```

边的 ICP 初值从初始全局位姿推导：`Z0 = T_target^{-1} T_source`。
`edgeStatuses` 对每条调用者指定的边返回接受状态、迭代数、配对数、RMS 和原因。

## 算法说明

每一轮迭代：

1. 用当前变换 `T` 变换全部源点 `x' = R x + t`。
2. 在**只构建一次、各轮复用**的目标 KD 树上为每个变换后源点找最近目标点。
   允许多个源点匹配同一目标；等距时取**目标原始索引较小者**（结果确定）。
3. 剔除欧氏距离大于 `maxCorrespondenceDistance` 的配对；不保存完整距离矩阵，
   KD 树最近邻为 O(log n) 期望复杂度。
4. 对保留配对求最小二乘刚体增量 Δ（Horn/Umeyama，质心去均值 + SVD），
   反射情形翻转 `U` 的最后一列以禁止镜像，并累计 `T ← Δ·T`。
5. 平移增量 `‖Δt‖` 与旋转增量 `acos((tr(ΔR)−1)/2)` **同时**低于各自阈值才收敛。
6. 输出指标（配对数、RMS）一律用**最终变换重新匹配**计算，不沿用上一轮误差。

### 失败语义（不任取变换冒充成功）

| 场景 | 行为 |
| --- | --- |
| 点云为空 / 含非有限坐标 / 配置非正 / 初值非刚体 | 抛 `std::invalid_argument` |
| 有效配对少于 3 组 | `converged=false`，原因 `InsufficientCorrespondences` |
| 配对共线导致旋转不可唯一确定 | `converged=false`，原因 `DegenerateGeometry` |
| 达到最大迭代次数 | `converged=false`，原因 `MaxIterationsReached`，返回当前估计 |

### 数值容差

- 初始旋转校验：`max|RᵀR − I| ≤ 1e-9` 且 `|det R − 1| ≤ 1e-9`。
- 旋转可解性：去均值点集不能共线；判据为第二奇异值必须大于
  `rankEpsilon · max(1, σ₁)`（默认 `1e-10`）。非共线三点或平面点云秩为 2，
  仍可唯一确定刚体旋转，因此不会被误判退化。
- 收敛判据使用严格小于阈值；角度经 `acos` 前夹紧到 `[-1, 1]`。
- 输出旋转由 SVD 正交矩阵乘积给出，天然满足正交与行列式 1，不引入缩放。

### 多站联合优化

设站点全局位姿为 `T_i`，边测量为源站到目标站的相对刚体变换 `Z_ij`：

```text
P_ij = T_j^{-1} T_i
r_ij = log_SE3(Z_ij^{-1} P_ij)
```

残差切向量顺序为 `[平移; 轴角]`，平移、旋转分别乘调用者给定正权重。相邻边使用
平方损失；闭环边对加权残差范数使用 Huber 损失，阈值由 `huberDelta` 配置。
优化在 SE3 切空间更新并经指数映射回刚体流形，首站固定，其余站点联合求解。
更新必须不增加总目标；数值求解失败或找不到可接受步长时保留最后有效估计。

| 多站场景 | 行为 |
| --- | --- |
| 非法位姿、非有限点、无效站号、自环、非法参数 | 抛 `std::invalid_argument` |
| 指定边 ICP 未收敛或退化 | 标记拒绝并给出逐边原因，测量不入图 |
| 有效边无向图不连通 | 返回 `GraphDisconnected`，不固定漂浮分量 |
| 达到最大迭代 | 返回 `MaxIterationsReached` 和当前估计 |
| 数值失败 | 返回 `NumericalFailure` 和最后有效估计 |

## 示例输出摘要

`make example` 使用 80 个已知变换采样点 + 2 个远距离离群点、240 个目标点：

```
配准前最近点 RMS: 9.686369
终止原因: Converged（已收敛）
迭代次数: 30, 最终有效配对数: 80（离群点已被距离阈值剔除）
配准后最近点 RMS（最终变换重新匹配）: 6.54e-15
```

四站闭环示例展示逐站拼接漂移和闭环联合校正结果：

```text
校正前各站位姿误差合计: 0.782269
校正前闭环平移不闭合量: 0.398497
...
终止原因: Converged（成功） - 目标函数下降量低于收敛容差
迭代次数: 4
目标值: 0.0633497 -> 1.92707e-16
校正后各站位姿误差合计: 3.44701e-08
校正后闭环平移不闭合量: 9.78892e-09
融合点云点数: 57
```

## 测试覆盖

`tests/test_icp.cpp` 覆盖：KD 树正确性与等距 tie-break、SVD 刚体恢复、
非共线平面三点恢复、正交/行列式保证、配对不足与共线退化、SE3 exp/log/逆变换、
闭环位姿图联合校正、ICP 收敛与离群点剔除、多站逐边原因、有效图不连通、
重复调用隔离、空集/非有限值/非法参数/自环校验和输入不可变。
