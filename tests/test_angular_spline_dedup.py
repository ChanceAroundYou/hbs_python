"""`angular_welding` 只拟合一次样条，求解器与极点搜索共用同一个 `_SplineShape`。

2026-10-30 之前 `angular_welding` 走 `compute_direct_hbs_from_boundary(bound)`，
内部就拟合一次周期三次样条并搜一次极点；随后又构造 `_SplineShape(bound)` 再拟合
一遍，供下游的 `shape.curve(parameter)` 使用。两次拟合的节点相同，所以数值一致，
但同一条曲线被拟合两次、极点也搜两次。

现在求解器直接吃 `shape.curve` / `shape.derivative`，极点从 `shape.points` 算。
`_SplineShape` 内部强制逆时针；`prepare_boundary` 已把输入统一翻成顺时针，所以
这里恰好翻一次回到 CCW。
"""

from __future__ import annotations

import numpy as np

from hbs._angular import direct
from hbs._pipeline import prepare_boundary
from hbs.angular import angular_welding


def _ellipse(n: int = 1024) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([1.3 * np.cos(t), 0.8 * np.sin(t)], axis=1)


def test_solver_takes_the_curve_and_the_point_path_is_unused(monkeypatch):
    """给求解器的是解析曲线句柄；点集入口（会再拟合一次样条）不再被调用。"""
    seen = []
    real = direct.compute_direct_hbs

    def spy(curve, derivative, n=480, pole=0j):
        seen.append((curve, derivative, n, pole))
        return real(curve, derivative, n=n, pole=pole)

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "angular_welding 不应再走 compute_direct_hbs_from_boundary"
        )

    monkeypatch.setattr(direct, "compute_direct_hbs", spy)
    monkeypatch.setattr(direct, "compute_direct_hbs_from_boundary", forbidden)

    angular_welding(
        prepare_boundary(_ellipse(), 300, resample=False), circle_num=300, n=512
    )

    assert len(seen) == 1
    curve, derivative, n, pole = seen[0]
    assert callable(curve) and callable(derivative)
    assert n == 512
    assert abs(pole) < 1.0


def test_reversed_input_gives_the_same_welding():
    """输入遍历方向翻转后焊接逐位不变：样条按弧长均匀配点，旋转与翻转被规范吸收。"""
    prepared = prepare_boundary(_ellipse(), 300, resample=False)
    forward = angular_welding(prepared, circle_num=300, n=512)
    backward = angular_welding(prepared[::-1], circle_num=300, n=512)
    for key in ("init_x", "init_y", "parameter"):
        assert np.abs(forward[key] - backward[key]).max() < 1e-12
