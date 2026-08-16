# HBS（Harmonic Beltrami Signature）— hbs_python

数学库：图像 → 边界 → HBS（Beltrami 系数复向量，定义在三角面心）→ LSQC 重建。
发布为 PyPI 包 `hbs`（当前 1.1.0，GitHub `ChanceAroundYou/hbs_python`）。
下游消费方：hbs_generation（VAE 在 HBS 空间生成形状，独立仓库）。

## 模块地图

```
hbs/__init__.py            公共命名空间（见下）
hbs/hbs.py                 get_hbs / reconstruct_from_hbs（主流程）
hbs/mesh.py                Mesh / DiskMesh / get_rect / get_unit_disk / get_unit_disk_in_rect
hbs/qc/beltrami.py         get_beltrami_coefficient / mu_chop（Beltrami 系数与截断）
hbs/qc/lsqc.py             lsqc_solver / unsigned_area_matrix / div_PtP_grad
hbs/qc/bc.py               shim → hbs.qc.beltrami（back-compat）
hbs/conformal_welding.py   ConformalWelding / get_conformal_welding
hbs/utils/boundary.py      get_boundary（需 opencv，extras `hbs[boundary]`）
hbs/utils/cast.py          to_real / to_complex
hbs/utils/poisson.py       调和扩张 Poisson 积分
hbs/utils/geodesic_welding.py / mobius.py / zipper.py
hbs/utils/tool_functions.py shim → hbs.utils.cast + hbs.qc.beltrami
hbs/plotting.py            可选绘图（matplotlib 惰性 import，extras `hbs[plot]`）
tests/                    pytest 回归基座（26 用例，锁定行为）
```

README.md 有完整结构/安装/extras。

## 公共 API

`import hbs` 暴露：`get_hbs`、`reconstruct_from_hbs`、`get_beltrami_coefficient`、
`lsqc_solver`、`mu_chop`、`get_conformal_welding`、`get_unit_disk`、`get_rect`、
`get_unit_disk_in_rect`、`Mesh`、`DiskMesh`。

- `get_hbs(bound, circle_point_num=500, density=0.01, disk=None)` → `(hbs, hbs_mapping, cw, disk)`
  - `bound`：n×2 float 数组，**方向任意——内部统一翻转为顺时针**（math 约定 signed area<0）
  - 过疏边界（<400 点）自动重采样到 500
- `reconstruct_from_hbs(hbs, disk, eps=0.0)` → `(bound_points, in_points, out_points, mapping)`
- `lsqc_solver(mu, landmark, target, mesh, eps=0.0)`：landmark 顶点索引 → target 位置约束
- 输入/输出 dtype：`get_hbs` 返回 complex；`to_real`/`to_complex` 实↔复二维数组互转

## 数学与数值关键事实（易踩坑）

1. **|μ| 必须 < 1**。`mu_chop(bound=0.9999, eps=1e-6)` 光滑单调饱和保序
   （顶部梯度 0.9999/0.99995/1.0 → 0.99990/0.99994/0.99996，非旧硬墙）。
2. **μ 尺度不变 → 重建尺度由 landmark 定**。reconstruct 把 disk 顶点 (1,0)/(0,0) 钉到
   target (1,0)/(0,0)，强制重建形状 x 半轴 = 1.0，**面积比 = 1/rx²**（rx=原形状 x 半轴）。
   形状本身保真（`rec × rx ≈ 原始`，mean err < 0.01）。这是 μ 表示法的数学事实，
   **不是 seam/焊接病态**（早期误诊）。要保绝对尺度需在重建后乘回 rx。tests 已锁定该定律。
3. **`reconstruct_from_hbs` 必须与生成时同一个 `DiskMesh`**——face 数不匹配 LSQC 断言崩。
4. **HBS 长度约定**：`get_unit_disk(0.01, 1000)` → 62512 faces（下游 1D VAE 尾部补零到 65536）。
5. **边界分辨率鲁棒性**：500 是验证过的稳健默认；某些计数（250 及 ≥800）zipper 数值退化，
   静默退化输出需调用方校验重建面积。
6. **mesh 算子符号约定**：Dx@x = -1, Dy@y = -1（顺时针取向符号），交叉项 0。

## 依赖

硬依赖 numpy + scipy（核心计算无需 matplotlib）。extras：`hbs[plot]`（matplotlib）、
`hbs[boundary]`（opencv）、`hbs[all]`。

## 下游兼容红线（hbs_generation）

只用 4 个公开 API：`hbs.get_hbs`、`hbs.reconstruct_from_hbs`、`hbs.mesh.get_unit_disk`、
`hbs.utils.boundary.get_boundary`。back-compat shim 保旧路径不破
（`hbs.qc.bc`、`hbs.utils.tool_functions`）。改动不得破坏这两个 shim。

## 开发流程

```bash
.venv/bin/python -m pytest tests/ -q     # 26 passed（重构收尾期基座）

# 解耦门：核心 import 不得加载 matplotlib
.venv/bin/python -c "import subprocess,sys; c='import hbs,hbs.mesh,hbs.utils.boundary;print(\"matplotlib\" in sys.modules)'; print(subprocess.run([sys.executable,'-c',c],capture_output=True,text=True).stdout)"

# 下游冒烟（PYTHONPATH 指本地源码）
PYTHONPATH=. .venv/bin/python -c "from hbs import get_hbs, reconstruct_from_hbs; from hbs.mesh import get_unit_disk; from hbs.utils.boundary import get_boundary; print('downstream OK')"
```

- 提交用 Conventional Commits。
- **重构行为零变化原则**：算法逻辑改动前先加测试锁定行为；只移动/改名/删死代码不得改变数值结果。
