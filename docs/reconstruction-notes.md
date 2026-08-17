# HBS 重建优化 — 探索结论与设计记录

> 文档日期：2026-08-17。记录 λ-归一化/噪声鲁棒性的已交付修复与**凹角重建的定性与立项**，避免后续重复误诊。

## 1. 已交付（commit `2ec650f`，分支 `worktree-fix-hbs-robustness`）

### A1 — 噪声边界 get_hbs 恢复
- `get_hbs` 入口 `_validate_boundary`：拒绝退化/NaN/重复点/零周长/零面积，给清晰 `ValueError`。
- 新增 `ConformalWeldingNumericalError`，收敛 zipper 与归一化的数值失败，不再泄露 `argmin(empty)` 等内部错误。
- **仅在该数值失败时**对已重采样闭曲线做 `gaussian_filter1d(σ=2.0, mode="wrap")` 平滑重试一次；健康/尖角输入永不预先平滑（避免钝化真实角）。

### A2 — λ 归一化（`_lambda_phase`）
- 面积加权离散 `I₂ = Σ_face μ_f·|A_f|/z_f`（原实现漏乘面积，且 `isnan + norm==0` 因运算符优先级从未剔除 NaN 面片）。
- 保守近零相位门：`|I₂|/Σ|term| < 1e-4` 视为无可靠归一化方向，直接返回 GHBS 代表（不旋转）。
- 迭代上限 `30 → _LAMBDA_MAX_ITERATIONS = 8`（实验收敛：椭圆 3 次、花生 5 次）。
- 精确对称轨道（Z3）理论上不存在连续唯一 λ 代表（见 HBSN `lambda_normalization_notes.tex`），不承诺收敛为独特角度，只保证不被离散残差驱动伪旋转。

### 验证
- 全套 `58 passed + 1 xfail`（既有 Z3 三角形已知局限）；`import hbs` 不加载 matplotlib；下游 4 API 冒烟 OK。
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
- `hbs/hbs.py`：`get_hbs` / `_lambda_phase` / `reconstruct_from_hbs` / `_validate_boundary`
- `hbs/conformal_welding.py`：`ConformalWeldingNumericalError` / `get_conformal_welding`
- `tests/test_hbs_robustness.py`
- 理论：`/home/nnb/projects/HBSN/latex/internal/lambda_normalization_notes.tex`、`hbs_normalization_theory.tex`；论文 `refs/Lin_Lui_2022_HarmonicBeltramiSignature.pdf` §6.5（论文自承 recessed border 会丢，建议加大 N；本库证实纯 hbs 加大 N 无效，因信息已丢）。
