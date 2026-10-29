# HBS 重建优化 — 探索结论与设计记录

> 文档日期：2026-08-17。记录 λ-归一化/噪声鲁棒性的已交付修复与**凹角重建的定性与立项**，避免后续重复误诊。

## 1. 已交付（commit `2ec650f`，分支 `worktree-fix-hbs-robustness`）

### A1 — 噪声边界 get_hbs 恢复
- `get_hbs` 入口 `_validate_boundary`：拒绝退化/NaN/重复点/零周长/零面积，给清晰 `ValueError`。
- 新增 `ConformalWeldingNumericalError`，收敛 zipper 与归一化的数值失败，不再泄露 `argmin(empty)` 等内部错误。
- **仅在该数值失败时**对已重采样闭曲线做 `gaussian_filter1d(σ=2.0, mode="wrap")` 平滑重试一次；健康/尖角输入永不预先平滑（避免钝化真实角）。

### A2 — I₂ 相位规范（2.1.0 起一次性施加，无迭代）
- 面积加权离散 `I₂ = Σ_face μ_f·|A_f|/z_f`（原实现漏乘面积，且 `isnan + norm==0` 因运算符优先级从未剔除 NaN 面片）。
- 保守近零相位门：`|I₂|/Σ|term| < 1e-4` 视为无可靠归一化方向，直接返回 GHBS 代表（不旋转）。
- **2.1.0 变更**：曾有 `_LAMBDA_MAX_ITERATIONS = 8` 的迭代（`rotate_x(r)` → `linear_interp`
  → 重算场，逼近 arg I₂ = 0）。`rotate_x(r)` 转的是源圆参数化，而随后的 `linear_interp`
  总是重采样到从角 0 起的固定 x 网格 —— r 不是步长整数倍时它不是刚体旋转，停止门留下的
  残余随输入起点变化（seam 处 dμ/dr 增益实测 ~125×）。该循环已删除，改为一次性共轭旋转
  `μ ← e^{-2iθ}μ` 且网格顶点同步乘 `e^{-iθ}`（`hbs.gauges.apply_i2`），不重解、不插值。
- 精确对称轨道（Z3）理论上不存在连续唯一 λ 代表（见 HBSN `lambda_normalization_notes.tex`），不承诺收敛为独特角度，只保证不被离散残差驱动伪旋转。

### A3 — 论文 M3+M4 规范与 `auto` 切换（2.3.0；**`auto` 已于 2.3.2 退役**）

> 现行默认是 `normalize="default"`（= 无分支锚 `c0`）。`"auto"` 不再被接受；
> `"i2"`/`"m34"` 仍可显式选择。本节保留 2.3.0 的设计记录。

- 新增 `hbs.gauges.select_gauge(mu, disk, kind)`，2.3.0 时 `kind ∈ {"auto","i2","m34","none"}`。
  `"m34"` 是论文 (5.8)+(5.9) 的**闭式两分支**：`τ₀ = arg ∫μ dA`、`τ₁ = arg ∫μ/z dA`、
  `s = (τ₁ − τ₀/2) mod 2π`，`s ∈ [0,π)` 取 `T = τ₀/2`，否则 `T = τ₀/2 + π`。
- `"auto"`（默认）在 `|I₂|/Σ|term| ≥ AUTO_I2_CONFIDENCE = 1e-2` 时用 I₂，否则退到
  M3+M4；两者都不可信时**不旋转**，报 `GaugeChoice.kind == "none"`。
- 旋转作用律实测：`I₂` 权重 1（`→ e^{-iT}`）、`M₃` 权重 2（`→ e^{-2iT}`）。权重差正是
  M3+M4 需要两个泛函的根据：M₃ 的权重 2 只能定出 `T mod π`，I₂ 的权重 1 挑分支。
- 分支翻转（`T → T+π`）在重建意义上无害：`e^{-2i(T+π)} = e^{-2iT}`，`μ` 逐位不变、重建
  fid ~1e-14；但 `disk` 顶点会转半圈、`gauge.phase` 会跳 π，按相位或按顶点做跨形状比较的
  下游会看到不连续。这是 `auto` 让 I₂ 优先的理由。
- `HBSResult.gauge` 交回 `GaugeChoice`（`kind` / `phase` / `confidence_i2` /
  `confidence_m3` / `branch` / `branch_margin`）。
- **`AUTO_I2_CONFIDENCE` 未经扫描优化**：实测的六个旋转不对称形状 I₂ 可信度均在
  2.7%–5.7%，全部高于门限，因此该批样本上 `auto` ≡ `i2`；兜底分支只在近对称形状
  （如三齿 0.04%）上触发过。

### 验证
- 全套 `143 passed + 2 skipped + 1 xfail`（2.3.1 口径；跳过项为缺 cv2，xfail 为既有 Z3 三角形已知局限）；`import hbs` 不加载 matplotlib；下游 4 API 冒烟 OK。
- 整段独立 review：Spec ✅ / Quality ✅(COMMENT) / 0 个 HIGH+CRITICAL / APPROVE。

## 2. 凹角/星形重建 — 定性与立项

**核心结论：凹角/星形几何无法仅凭 `hbs` 恢复。**

证据链（试验实证）：
- 自由边界 LSQC/P1-FEM 映射把任意形状的 seam 塌缩成半径≈1 的近正圆。star5 的 LSQC seam 半径 std≈0.00066——星形的 0.6/1.0 径向波动彻底丢失。
- 纯 LSQC 全局解（内域）、外域矩形全局解、焊接管线三者 star fid 均≈0.18-0.34（同源塌缩）。
- 仅当把**真实星形 seam 作为全局 Dirichlet 约束**时，star 才完美恢复（fid≈0.0001）——但真实 seam 是重建的未知输出，运行时不可得。

已实测定死、勿再走的路径：
1. **线性加密 seam/welding 点**：真实 `circle_point_num=1000` 下 star5 保真度≈0.362 仍塌缩。线性插值只在已压缩区间内加点，不恢复丢失信息。
2. **get_hbs 额外保存 seam + reconstruct 接收**：等价于把原图抄进重建再声称能重建（被用户否决为循环自证）。`reconstruct_from_hbs` 必须仅凭 hbs 工作。

**立项（待做）**：凹角恢复需换掉共享的 LSQC 自由边界阶段。候选：A) 独立 QC/Beltrami 全局求解器（边界约束）；B) 外部共形映射/QC-map 库；C) P1 元角点奇异 enrichment（共享根因，风险大，不优先）。

**验收指标**：用**旋转对齐**保真度（纯 cyclic 对齐不够，λ 输出合法旋转）。star5 目标 <0.15（现纯 hbs ≈0.1769）。

## 3. 相关文件
- `hbs/hbs.py`：`get_hbs` / `_prepare_boundary` / `_canonical_start_index` / `reconstruct_from_hbs` / `_validate_boundary`
- `hbs/gauges.py`：`i2_phase` / `apply_i2` / `rotate_disk`（两条路径共用的相位规范）
- `hbs/conformal_welding.py`：`ConformalWeldingNumericalError` / `get_conformal_welding`
- `tests/test_hbs_robustness.py`
- 理论：`/home/nnb/projects/HBSN/latex/internal/lambda_normalization_notes.tex`、`hbs_normalization_theory.tex`；论文 `refs/Lin_Lui_2022_HarmonicBeltramiSignature.pdf` §6.5（论文自承 recessed border 会丢，建议加大 N；本库证实纯 hbs 加大 N 无效，因信息已丢）。
