# HBS（Harmonic Beltrami Signature）— hbs_python

数学库：图像 → 边界 → HBS（Beltrami 系数复向量，定义在三角面心）→ LSQC 重建。
发布为 PyPI 包 `hbs`（当前 2.3.1，GitHub `ChanceAroundYou/hbs_python`）。
下游消费方：hbs_generation（VAE 在 HBS 空间生成形状，独立仓库）。

## 模块地图

```
hbs/__init__.py            公共命名空间（见下）
hbs/_engines.py            两套独立引擎 + 注册表（`ENGINES`）；互不替补
hbs/_pipeline.py           焊接之后的唯一后端：准备/延拓/μ/I₂ + `compare_engines`
hbs/hbs.py                 get_hbs / get_hbs_result / reconstruct_from_hbs
hbs/angular.py             角速度引擎（边界层对数单层势）→ `angular_welding`
hbs/_angular/direct.py     角速度求解内核（vendored，只依赖 numpy/scipy）
hbs/_angular/joint.py      联合全场上游实现（vendored；库代码已不用，见 CHANGELOG）
hbs/mesh.py                Mesh / DiskMesh / get_rect / get_unit_disk / get_unit_disk_in_rect
hbs/qc/beltrami.py         get_beltrami_coefficient / mu_chop（Beltrami 系数与截断）
hbs/qc/lsqc.py             lsqc_solver / unsigned_area_matrix / div_PtP_grad
hbs/conformal_welding.py   ConformalWelding / get_conformal_welding
hbs/extension.py           FEM 调和延拓（两引擎共用）
hbs/cache.py               网格 + FEM 算子的磁盘缓存
hbs/gauges.py              源圆旋转规范（I₂ 单锚 / 论文 M3+M4 闭式，auto 切换）
hbs/weld_compare.py        焊接一致性度量（次数 1 圆同胚 + 源圆旋转对齐）
                           → tests/test_forward_invariance.py 钉前向不变性契约
hbs/utils/boundary.py      get_boundary（需 opencv，extras `hbs[boundary]`）
hbs/utils/cast.py          to_real / to_complex
hbs/utils/poisson.py       稠密 Poisson 延拓（**仅实验对照**，库代码不用）
hbs/utils/geodesic_welding.py / mobius.py / zipper.py
hbs/utils/eval.py          相似不变保真度 fidelity_sim
hbs/plotting.py            可选绘图（matplotlib 惰性 import，extras `hbs[plot]`）
unpublished_gpl/            GPL 隔离区（不在包内、不进 wheel，见下）
tests/                    pytest 回归基座（103 passed + 2 skipped + 1 xfailed）
```

README.md 只写现行行为（结构/安装/用法/extras）；求解器对照、实测基准与
2.0.x 变更记录在 CHANGELOG.md。

## 公共 API

`import hbs` 暴露：`get_hbs`、`get_hbs_result`、`reconstruct_from_hbs`、
`compare_engines`、`get_beltrami_coefficient`、`lsqc_solver`、`mu_chop`、
`get_conformal_welding`、`compare_welding`/`welding_curve`、
`get_unit_disk`、`get_rect`、`get_unit_disk_in_rect`、`Mesh`、`DiskMesh`。

- `get_hbs(bound, circle_point_num=500, density=0.01, disk=None,
  method="angular", normalize="auto")` → `(hbs, hbs_mapping, cw, disk)`
  - `bound`：n×2 浮点数组，绕向任意（内部统一翻到顺时针）
  - `method` 真的切换引擎，两个引擎**互不替补**：
    - `"angular"`（默认）：边界层对数单层势 → `hbs.angular.angular_welding`
    - `"zipper"`：Bishop/Marshall zipper → `zipper_welding_prepared`
  - 焊接之后（并行延拓、μ、I₂）完全共用 `hbs._pipeline`，只有一份实现
  - `normalize`：`"auto"`（默认）/`"i2"`/`"m34"`/`"none"`，语义见 `hbs.gauges`
    模块文档。`auto` 在 `|I₂|/Σ|term| ≥ 1e-2` 时用 I₂，否则退到论文 M3+M4；
    切换**不连续**，诊断量在 `HBSResult.gauge`。`"first-moment"` 已在 2.2.0 移除
  - 两个引擎的焊接只差源圆旋转：`compare_engines` 度量，N=500 时
    `welding_rms = 0.08°`，`mu_rms = 0.006`（`mu_max ≈ 0.39` 由边界层主导）
  - `get_conformal_welding(bound, method=...)` 走同一引擎注册表，对同一 `method`
    给出与 `get_hbs` 内部**逐位相同**的 `(x, y)`
- `get_hbs_result(...)` → `HBSResult(mu, field, welding, disk, engine, rung, gauge)`：
  `rung` 非 `"raw"` 表示原始输入失败、用了降级的平滑重采样；`gauge` 是
  `GaugeChoice(kind, phase, confidence_i2, confidence_m3, branch, branch_margin)`
- `reconstruct_from_hbs(hbs, disk, eps=0.0)`
  → `(bound_points, in_points, out_points, mapping)`
- `lsqc_solver(mu, landmark, target, mesh, eps=0.0)`：landmark 顶点索引 → target 位置约束
- 输入/输出 dtype：`get_hbs` 返回 complex；`to_real`/`to_complex` 实↔复二维数组互转

## 每个引擎自己的输入阶梯（2.2.0：每引擎独立）

两个引擎各自维护自己的阶梯（`hbs._engines._rungs`），失败时**只在自己内部**降级，
绝不切换到另一个引擎。角速度引擎要求光滑 Jordan 边界；像素阶梯轮廓
（`cv2.findContours`）与噪声轮廓会让原始调和测度密度变负、直接失败。此时按

```
raw → 弧长重采样500 → 重采样500+σ2 高斯 → 重采样500+σ4 高斯
```

逐级重试，第一个成功的结果即返回，落点报在 `HBSResult.rung`（非 `"raw"` 即降级）。
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
   始终复用 `get_hbs` 返回的 `disk`：I₂ 规范会整体旋转顶点，返回的是**新**对象。
4. **HBS 长度约定**：默认网格 `get_unit_disk(0.01, 500)` → 60796 faces；
   `get_unit_disk(0.01, 1000)` → 62512 faces（下游 1D VAE 尾部补零到 65536）。
   2.1.0 起两个引擎共用这一个默认网格，不再有 angular 专用的 `(0.04, 512)`。
5. **边界分辨率鲁棒性**：重建点数由 `disk.circle_num`（默认 500）决定；
   zipper 对某些计数（250 及 ≥800）数值退化。
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
