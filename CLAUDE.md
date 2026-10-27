# HBS（Harmonic Beltrami Signature）— hbs_python

数学库：图像 → 边界 → HBS（Beltrami 系数复向量，定义在三角面心）→ LSQC 重建。
发布为 PyPI 包 `hbs`（当前 2.0.2，GitHub `ChanceAroundYou/hbs_python`）。
下游消费方：hbs_generation（VAE 在 HBS 空间生成形状，独立仓库）。

## 模块地图

```
hbs/__init__.py            公共命名空间（见下）
hbs/hbs.py                 get_hbs / reconstruct_from_hbs（主流程）
hbs/angular.py             角速度前向求解器（2.0.0 默认）+ 失败时输入阶梯
hbs/_angular/direct.py     角速度求解引擎（vendored，只依赖 numpy/scipy）
hbs/_angular/joint.py      调和延拓落场引擎（vendored，只依赖 numpy/scipy）
hbs/mesh.py                Mesh / DiskMesh / get_rect / get_unit_disk / get_unit_disk_in_rect
hbs/qc/beltrami.py         get_beltrami_coefficient / mu_chop（Beltrami 系数与截断）
hbs/qc/lsqc.py             lsqc_solver / unsigned_area_matrix / div_PtP_grad
hbs/qc/bc.py               shim → hbs.qc.beltrami（back-compat）
hbs/conformal_welding.py   ConformalWelding / get_conformal_welding
hbs/utils/boundary.py      get_boundary（需 opencv，extras `hbs[boundary]`）
hbs/utils/cast.py          to_real / to_complex
hbs/utils/poisson.py       调和扩张 Poisson 积分
hbs/utils/geodesic_welding.py / mobius.py / zipper.py
hbs/utils/eval.py          相似不变保真度 fidelity_sim
hbs/utils/tool_functions.py shim → hbs.utils.cast + hbs.qc.beltrami
hbs/plotting.py            可选绘图（matplotlib 惰性 import，extras `hbs[plot]`）
unpublished_gpl/            GPL 隔离区（不在包内、不进 wheel，见下）
tests/                    pytest 回归基座（96 passed + 1 xfailed）
```

README.md 只写现行行为（结构/安装/用法/extras）；求解器对照、实测基准与
2.0.x 变更记录在 CHANGELOG.md。

## 公共 API

`import hbs` 暴露：`get_hbs`、`angular_mu_from_boundary`、`reconstruct_from_hbs`、
`get_beltrami_coefficient`、`lsqc_solver`、`mu_chop`、`get_conformal_welding`、
`get_unit_disk`、`get_rect`、`get_unit_disk_in_rect`、`Mesh`、`DiskMesh`。

- `get_hbs(bound, circle_point_num=500, density=0.01, disk=None, method="angular")`
  → `(hbs, hbs_mapping, cw, disk)`
  - `bound`：n×2 float 数组，方向任意（两条路径各自内部处理）
  - `method="angular"`（2.0.0 默认）：固定网格 `(density=0.04, circle=512)`，
    **忽略 `circle_point_num`/`density`**（显式传非默认值且未给 `disk` 时 `UserWarning`），
    返回 `cw=None`；显式 `disk=` 原样使用，不做 512→256 网格回退
  - `method="zipper"`（焊接路径）：`circle_point_num`/`density` 生效，边界一律按弧长
    重采样到 500，返回 `ConformalWelding`，λ-归一化最多 8 轮。**不保证数值复现 1.1.0**：
    归一化已由 `angle(Σμ)`+`rotate_x(r/2)`+30 轮换成面积加权 `I₂` 相位+`rotate_x(r)`+8 轮，
    seam 落点不同（椭球/花生类形状重建保真差 10–100×）。要复现 1.1.0 读数须装 `hbs==1.1.0`
- `reconstruct_from_hbs(hbs, disk, eps=0.0)`
  → `(bound_points, in_points, out_points, mapping)`
- `lsqc_solver(mu, landmark, target, mesh, eps=0.0)`：landmark 顶点索引 → target 位置约束
- 输入/输出 dtype：`get_hbs` 返回 complex；`to_real`/`to_complex` 实↔复二维数组互转

## 角速度路径的失败时输入阶梯（2.0.0）

角速度求解要求光滑 Jordan 边界。像素阶梯轮廓（`cv2.findContours`）与噪声轮廓会让
原始调和测度密度变负、直接失败。此时**且仅在此时**按

```
raw → 弧长重采样500 → 重采样500+σ2 高斯 → 重采样500+σ4 高斯
```

逐级重试，第一个通过重建合法性门的结果即返回，并标记
`stats["boundary_rung"]` / `stats["input_smoothed"]`（非 `"raw"` 时发 `UserWarning`）。
原始成功时阶梯不启动，结果与无阶梯完全一致。全部失败 → `RuntimeError`（`__cause__`
保留最后一级原始错误），**不返回退化曲线**。

MPEG-7 抽样实测 [X]（真实 cv2 轮廓；可用性 = 重建有限非单点 且 尺度/方向无关形状
保真 < 0.30）：阶梯设计探针 95 例 raw 68 → 阶梯 72（zipper 69）；装成 wheel 后
用包内实现复核 55 例 raw 37 → 阶梯 39（zipper 38）。被救回例的保真 0.049–0.157，
明显劣于原始输入成功时的中位 0.0019 ——「可用但退化」，故必须标记，不得当作等价结果。
两引擎都拒绝的比例：探针 22/95、复核 16/55。

## 数学与数值关键事实（易踩坑）

1. **|μ| 必须 < 1**。`mu_chop(bound=0.9999, eps=1e-6)` 光滑单调饱和保序
   （顶部梯度 0.9999/0.99995/1.0 → 0.99990/0.99994/0.99996，非旧硬墙）。
2. **μ 尺度不变 → 重建尺度由 landmark 定**。reconstruct 把 disk 顶点 (1,0)/(0,0) 钉到
   target (1,0)/(0,0)，强制重建形状 x 半轴 = 1.0，**面积比 = 1/rx²**（rx=原形状 x 半轴）。
   形状本身保真（`rec × rx ≈ 原始`，mean err < 0.01）。这是 μ 表示法的数学事实，
   **不是 seam/焊接病态**（早期误诊）。要保绝对尺度需在重建后乘回 rx。tests 已锁定该定律。
   推论：用「重建边界 ÷ 原边界的原始面积比」判可用性是错的（会得到 1e-6 量级），
   必须用尺度/方向无关的形状保真度。
3. **`reconstruct_from_hbs` 必须与生成时同一个 `DiskMesh`**——face 数不匹配 LSQC 断言崩。
   始终复用 `get_hbs` 返回的 `disk`（angular 下它可能是 256 回退网格）。
4. **HBS 长度约定**：`get_unit_disk(0.01, 1000)` → 62512 faces（下游 1D VAE 尾部补零到 65536）；
   angular 默认网格 `(0.04, 512)` → 4280 faces。
5. **边界分辨率鲁棒性**：angular 下重建点数由 `disk.circle_num`（默认 512）决定；
   zipper 下 500 是验证过的稳健默认，某些计数（250 及 ≥800）数值退化。
   **seam 旋转不是规范（gauge）**：旋转 seam 会改变重建本身，不只是整体旋转
   （相似不变残差 0.13–0.55，噪声量级 1e-12）。因此 λ-归一化的落点是需要
   验证的求解选择，不能当作「可任意选定的表示」——这也是 `method="zipper"`
   换归一化后数值不再等于 1.1.0 的原因。
6. **mesh 算子符号约定**：Dx@x = -1, Dy@y = -1（顺时针取向符号），交叉项 0。
7. **包必须自足**：包内不得 import 研究仓库的顶层模块（`direct_boundary_hbs`、
   `joint_hbs_optimizer`、`upgraded_hbs`）——源码树里它们可导入会掩盖 wheel 断裂。
   `tests/test_packaging_selfcontained.py` 锁定这条。
8. **3.8/3.9 的注解陷阱**：没有 `from __future__ import annotations` 的文件里，
   `np.ndarray[...]`、`list[int]`、`X | None` 都会在 import 期求值并抛 `TypeError`。
   已发布 1.1.0 因此在声明支持的 3.8 上**直接 import 失败**。同一测试用 AST 锁住三类。
9. **`import hbs` 必须零警告**：下游 `pytest -W error` 会被任何 import 期警告打挂。
   模块级 `warnings.warn` 的实验模块不得被包的顶层 import。
10. **GPL 代码不得进包**：`angle_slit_welding.py` 是 Marshall zipper 程序的 GPL 逐字
    移植，已隔离在 `unpublished_gpl/`（非包目录）。`find_packages()` 不收它，
    但若有人放回 `hbs/utils/` 会静默进 wheel —— 测试里有守卫。

## 依赖

硬依赖 numpy + scipy（核心计算无需 matplotlib）。extras：`hbs[plot]`（matplotlib）、
`hbs[boundary]`（opencv）、`hbs[all]`。

## 下游兼容红线（hbs_generation）

只用 4 个公开 API：`hbs.get_hbs`、`hbs.reconstruct_from_hbs`、`hbs.mesh.get_unit_disk`、
`hbs.utils.boundary.get_boundary`。back-compat shim 保旧路径不破
（`hbs.qc.bc`、`hbs.utils.tool_functions`）。改动不得破坏这两个 shim。

## 开发流程

```bash
.venv/bin/python -m pytest tests/ -q     # 96 passed, 1 xfailed

# 打包自足性（wheel 装到干净环境后默认路径可用；7 条，秒级）
.venv/bin/python -m pytest tests/test_packaging_selfcontained.py -q

# 解耦门：核心 import 不得加载 matplotlib
.venv/bin/python -c "import subprocess,sys; c='import hbs,hbs.mesh,hbs.utils.boundary;print(\"matplotlib\" in sys.modules)'; print(subprocess.run([sys.executable,'-c',c],capture_output=True,text=True).stdout)"

# 下游冒烟（PYTHONPATH 指本地源码）
PYTHONPATH=. .venv/bin/python -c "from hbs import get_hbs, reconstruct_from_hbs; from hbs.mesh import get_unit_disk; from hbs.utils.boundary import get_boundary; print('downstream OK')"
```

- 提交用 Conventional Commits。
- **重构行为零变化原则**：算法逻辑改动前先加测试锁定行为；只移动/改名/删死代码不得改变数值结果。
