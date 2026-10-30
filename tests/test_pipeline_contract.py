"""引擎 → 管线的契约：引擎只交「源圆均匀网格上的边界值」。

## 为什么有这组测试

2.3.3 及以前 `_pipeline.build_hbs` 自己调 `welding.linear_interp(circle_num)`，
数值路径因此依赖 `ConformalWelding` —— 一个按 zipper 语义定义的类（`angular`
构造出来的对象上 `x_post_norm` / `y_post_norm` / `uniquify` 直接
`NotImplementedError`），并且把「引擎的输出是否已经在均匀源角上」这条契约藏进
管线内部的一次调用里。若将来 angular 改成非均匀/自适应源角，那次**线性**插值会
静默吃掉新增精度。

重构后的契约：

- `ENGINES[method](bound, circle_num).values` 是长度 `circle_num` 的复数组，
  `|values| = 1`、绕圆一周严格单调、`values[0] == 1`（目标圆相位归零）；
- `build_hbs(values, welding_or_None, disk, fem, normalize)` 不做任何插值，也不
  需要 `ConformalWelding` 才能算 μ —— `welding` 只为 `rotate_x` 的记账服务。

前向数值与 2.3.3 逐位相同：μ、`field`、焊接的 `y`/`init_x`/`init_y`、盘顶点与
规范相位在 46 组配置（2 引擎 × 4 `normalize` × 4 形状，外加非默认网格、重建门与
`angular_mu_from_boundary` 入口）的 sha256 快照上逐一相同。唯一的例外是
`welding.x`——`rotate_x` 的修复有意改变它，见文件末尾那个测试。
"""

from __future__ import annotations

import numpy as np
import pytest

import hbs
from hbs._engines import ENGINES
from hbs._pipeline import build_hbs, prepare_boundary, resolve_disk
from hbs.conformal_welding import get_conformal_welding

CIRCLE = 120
DENSITY = 0.05


def _boundary(n: int = 300) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack(
        [
            1.2 * np.cos(t) + 0.09 * np.cos(2 * t + 0.7),
            np.sin(t) + 0.05 * np.sin(3 * t),
        ],
        axis=1,
    )


@pytest.fixture(scope="module")
def disk_fem():
    return resolve_disk(None, DENSITY, CIRCLE)


def _engine_result(engine: str):
    prepared = prepare_boundary(_boundary(), CIRCLE, resample=(engine != "angular"))
    return ENGINES[engine](prepared, CIRCLE)


@pytest.mark.parametrize("engine", ["angular", "zipper"])
def test_engine_values_live_on_the_uniform_source_grid(engine):
    """引擎交出的就是源圆均匀网格上的边界值，管线不再插值。"""
    result = _engine_result(engine)
    values = np.asarray(result.values)

    assert values.shape == (CIRCLE,), f"{engine}: {values.shape}"
    assert np.iscomplexobj(values)
    assert np.all(np.isfinite(values))
    assert np.allclose(np.abs(values), 1.0, rtol=0, atol=1e-12)

    # `linear_interp` 的输出约定：目标圆相位以首点归零，所以 values[0] 恰为 1。
    assert values[0] == 1.0, f"{engine}: 首点相位没有归零"

    steps = np.diff(np.unwrap(np.angle(values)))
    assert np.all(steps > 0), f"{engine}: y 不是严格单调"
    assert 0.98 * 2 * np.pi <= steps.sum() <= 2 * np.pi, (
        f"{engine}: 绕行总量 {steps.sum():.6f} 不是一周"
    )


@pytest.mark.parametrize("engine", ["angular", "zipper"])
def test_values_are_exactly_the_welding_y(engine):
    """`values` 不是另算的一份：它与 `welding.y` 逐位相同。"""
    result = _engine_result(engine)
    assert np.array_equal(np.asarray(result.values), np.asarray(result.welding.y))


@pytest.mark.parametrize("normalize", ["none", "i2", "m34", "default"])
def test_build_hbs_needs_no_welding_object(normalize, disk_fem):
    """μ 只依赖边界值：传 `welding=None` 与传焊接给出逐位相同的 μ / field / 网格。"""
    disk, fem = disk_fem
    result = _engine_result("angular")

    mu_a, field_a, welding_a, disk_a, gauge_a = build_hbs(
        result.values, result.welding, disk, fem, normalize
    )
    mu_b, field_b, welding_b, disk_b, gauge_b = build_hbs(
        result.values, None, disk, fem, normalize
    )

    assert welding_a is result.welding
    assert welding_b is None
    assert np.array_equal(mu_a, mu_b)
    assert np.array_equal(field_a, field_b)
    assert np.array_equal(disk_a.vert, disk_b.vert)
    assert gauge_a == gauge_b


def test_get_conformal_welding_and_get_hbs_share_the_resampled_welding():
    """两个公开入口交回的是同一个状态的焊接。

    修复前 `get_conformal_welding(normalize="none")` 跳过 `linear_interp`：
    zipper 会交回源角**非均匀**的原始焊接（实测相邻源角 0.021–0.052 rad），
    与 `get_hbs(normalize="none")` 的焊接不是同一件事。
    """
    bound = _boundary()
    for engine in ("angular", "zipper"):
        cw_forward = hbs.get_hbs(
            bound,
            method=engine,
            normalize="none",
            circle_point_num=CIRCLE,
            density=DENSITY,
        )[2]
        cw_direct = get_conformal_welding(
            bound, method=engine, normalize="none", resample_to=CIRCLE
        )
        assert np.array_equal(np.asarray(cw_forward.y), np.asarray(cw_direct.y)), engine
        assert np.array_equal(np.asarray(cw_forward.x), np.asarray(cw_direct.x)), engine


@pytest.mark.parametrize("engine", ["angular", "zipper"])
@pytest.mark.parametrize("normalize", ["none", "i2", "m34", "default"])
def test_returned_welding_agrees_with_the_mesh_and_the_field(engine, normalize):
    """规范之后 `welding.x` 必须仍是盘边界顶点，`welding.y` 必须仍是边界值。

    规范转的是**源圆**：`disk` 顶点与 `welding.x` 一起转，目标圆的值（`field`
    的边界值、`welding.y`）不动。

    修复前 `rotate_x` 转的是 `init_x_angle` 那张网格，而 `linear_interp` 之后
    `x` 已经在均匀源角上：angular 的原始网格本就均匀，两者等价；zipper 的不是
    （实测相邻源角 0.021–0.052 rad），于是 `x` 与盘顶点错开 3.1 rad，而
    `field[:circle_num]` 仍与 `y` 逐位配对。
    """
    result = hbs.get_hbs_result(
        _boundary(), method=engine, normalize=normalize,
        circle_point_num=CIRCLE, density=DENSITY,
    )
    n = result.disk.circle_num
    circle = result.disk.circle[:, 0] + 1j * result.disk.circle[:, 1]
    field_boundary = result.field[:n, 0] + 1j * result.field[:n, 1]

    assert np.allclose(np.asarray(result.welding.x), circle, rtol=0, atol=1e-12), (
        f"{engine}/{normalize}: x 与盘边界顶点的最大偏差 "
        f"{np.abs(np.asarray(result.welding.x) - circle).max():.2e}"
    )
    assert np.array_equal(field_boundary, np.asarray(result.welding.y))
