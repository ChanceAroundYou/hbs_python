"""`fidelity_sim` 回归门：相似不变保真度在健康形状上必须远低于 0.10。

注意不能用「重建面积 ÷ 原面积」判可用性 —— HBS 规范坐标系强制重建的 x 半轴
为 1.0，该比值恒在 1e-6 量级，与形状质量无关。见 CLAUDE.md「数学与数值关键事实」。
"""
import numpy as np

from hbs import get_hbs, reconstruct_from_hbs
from hbs.utils.eval import fidelity_sim


def _ellipse(rx=1.2, ry=1.0, n=500):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([rx * np.cos(t), ry * np.sin(t)], 1)


def test_circle_fid_sim_healthy(disk):
    shape = _ellipse(1.0, 1.0, 500)
    hbs, _, _, d = get_hbs(shape, 1000, 0.01, disk)
    rec, _, _, _ = reconstruct_from_hbs(hbs, d)
    assert fidelity_sim(rec, shape) < 0.10


def test_ellipse_fid_sim_healthy(disk):
    shape = _ellipse(1.2, 1.0, 500)
    hbs, _, _, d = get_hbs(shape, 1000, 0.01, disk)
    rec, _, _, _ = reconstruct_from_hbs(hbs, d)
    assert fidelity_sim(rec, shape) < 0.10
