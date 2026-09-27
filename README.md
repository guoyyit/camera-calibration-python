# Camera Calibration Service

相机内参标定与畸变校正服务。输入为带唯一帧标识的二维角点观测（多帧、可乱序、可重复、可损坏），输出主点、焦距、径向/切向畸变参数与每帧重投影误差。全部状态持久化在本地 SQLite，不依赖外部服务。

## 快速开始

```bash
uv sync
uv run uvicorn main:app          # 启动 HTTP 服务（存储路径用 CALIBRATION_DB 覆盖）
uv run pytest                    # 运行全部自动化测试
```

也可以直接以库方式使用：

```python
from calibration.service import CalibrationService

svc = CalibrationService("calib.db")
svc.create_task("my-task")
svc.submit_frame("my-task", "frame-001", object_points, image_points, (640, 480))
result = svc.calibrate("my-task")
undistorted = svc.undistort_image("my-task", image, border_value=0.0)
```

## 坐标约定

- **物方点**：标定板平面上的二维点（z = 0），使用标定板自身的度量坐标系，单位在同一任务内保持一致（米或毫米）。
- **像方点**：像素坐标，原点为左上角像素中心，x 向右、y 向下。
- **相机模型**：针孔模型 + Brown-Conrady 径向/切向畸变。归一化理想坐标 `(x_u, y_u)`，`r² = x_u² + y_u²`：

  ```
  x_d = x_u·(1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x_u·y_u + p2·(r² + 2·x_u²)
  y_d = y_u·(1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r² + 2·y_u²) + 2·p2·x_u·y_u
  u = fx·x_d + cx,   v = fy·y_d + cy
  ```

## 求解参数

| 参数 | 含义 |
|------|------|
| `fx, fy` | 焦距（像素单位） |
| `cx, cy` | 主点（像素坐标） |
| `k1, k2, k3` | 径向畸变系数 |
| `p1, p2` | 切向畸变系数 |

求解流程为 Zhang 氏方法：逐帧单应（归一化 DLT）→ 内参闭式解（零 skew 约束）→ 逐帧外参 → 全部参数联合 Levenberg 精化，最小化重投影误差。输出包括总体 RMS 重投影误差与**每帧** RMS 重投影误差、实际迭代次数与收敛标志。

**可配置项**（创建任务时或单次标定时指定）：

- `max_iterations`（默认 50）：最大迭代次数；
- `convergence_tolerance`（默认 1e-10）：收敛阈值（同时作用于 xtol/ftol/gtol）；
- `use_tangential` / `use_k3`：是否估计切向畸变与 k3 项。

## 帧的生命周期与有效性规则

每帧以 `frame_id` 唯一标识，**首次提交生效**：

- `accepted`：通过校验，参与标定；
- `duplicate`：`frame_id` 已存在，忽略（重试安全，不会重复应用）；
- `corrupt`：校验失败（形状非 (N,2)、点数 < 4、物像点数不一致、含 NaN/Inf、图像尺寸非正、点集共线退化），忽略；
- `abandoned`：被显式放弃，不再参与标定。

乱序、重复与损坏帧不会使标定结果偏离仅由有效帧计算得到的结果：求解前帧按 `frame_id` 排序，求解器对输入顺序不敏感。

## 增量标定

可随时追加观测并重新标定。由于求解是确定性的且帧按 `frame_id` 排序，**增量重标定的结果与相同有效观测集合的全量标定逐位一致**。已放弃帧不参与计算。

## 去畸变与正向映射

- `distort_points`：正向映射，理想（未畸变）像素坐标 → 观测（畸变）像素坐标；
- `undistort_points`：逆向映射，观测像素坐标 → 理想像素坐标（固定 20 次不动点迭代，保证确定性）；
- `undistort_image`：整图校正。对每个输出像素用正向映射求源位置并做双线性插值。

**边界像素策略（恒定填充）**：双线性插值的四个邻域像素中，任何落在源图像边界外的像素以常量 `border_value`（默认 0）参与加权，与 OpenCV `BORDER_CONSTANT` 语义一致。

## 持久化与崩溃恢复

- 所有状态存于本地 SQLite（WAL 模式，`synchronous=FULL`）。
- 每次变更在单个事务内完成：崩溃时要么整体提交、要么整体回滚，**不会留下部分写入**。
- 崩溃后重新打开即恢复至**最近已提交状态**；帧插入按 `(task_id, frame_id)` 幂等（`INSERT OR IGNORE`），重试不会重复应用观测。

## 并发模型

同一任务上的创建、追加、读取与重新标定通过**按任务的锁**串行化，所有并发操作可线性化：每个操作观察到的是一个全局操作序列产生的状态。不同任务之间并行。并发工作负载结束后，最终标定结果与串行执行一致且确定。

## 可复现性与数值容差

**可复现条件**：相同的有效观测集合（与提交顺序无关）、相同的配置、相同的依赖版本（numpy/scipy），多次标定输出**逐位一致**（测试中以 `np.array_equal` 断言）。求解器内部按 `frame_id` 排序，所有数值步骤（SVD、LM 迭代）均为确定性算法，无随机源。

**数值容差**：

- 合成无噪声数据上，内参恢复误差 < 1e-6 像素/系数，RMS 重投影误差 < 1e-8；
- 点映射往返（`undistort_points∘distort_points`）误差 < 1e-6 像素；
- 跨平台/跨 BLAS 实现可能存在末位浮点差异，建议以 1e-9 相对容差比较。

**精度限制**：

- 至少需要 2 帧、每帧至少 4 个非共线对应点；视角退化（各帧光轴近乎平行、标定板覆盖画面区域过小）会放大主点与畸变参数的不确定性；
- 畸变逆映射为迭代近似，残余误差约 1e-13 像素量级（固定迭代次数保证确定性）；
- 标定精度上限受输入角点定位精度限制，重投影 RMS 即该噪声水平的直接反映。

## HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/tasks/{task_id}` | 创建任务（幂等），可带 `config` |
| GET | `/tasks/{task_id}` | 任务状态与结果 |
| POST | `/tasks/{task_id}/frames` | 提交一帧观测，返回帧状态 |
| POST | `/tasks/{task_id}/frames/{frame_id}/abandon` | 放弃一帧 |
| POST | `/tasks/{task_id}/calibrate` | 标定（可临时覆盖配置） |
| GET | `/tasks/{task_id}/result` | 读取已存标定结果 |
| POST | `/tasks/{task_id}/undistort-points` | 观测像素 → 理想像素 |
| POST | `/tasks/{task_id}/distort-points` | 理想像素 → 观测像素 |

## 项目结构

```
calibration/
  models.py     # 数据模型与坐标约定
  solver.py     # Zhang 氏标定求解器（确定性）
  image_ops.py  # 去畸变 / 正向映射 / 整图校正
  storage.py    # SQLite 持久化（事务、崩溃恢复）
  service.py    # 校验、按任务加锁、编排
  api.py        # FastAPI 路由
tests/          # 乱序重复帧、增量重标定、崩溃恢复、并发竞争、边界像素等测试
```
