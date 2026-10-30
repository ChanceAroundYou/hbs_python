from __future__ import annotations

import numpy as np
import warnings

from scipy.interpolate import interp1d

from hbs.utils.mobius import mobius, mobius_inv
from hbs.utils.cast import to_complex
from hbs.utils.zipper import zipper, zipper_params


class ConformalWeldingNumericalError(RuntimeError):
    """The zipper pipeline cannot produce a usable circle correspondence."""


class ConformalWelding:
    def __init__(
        self,
        x: np.ndarray[np.complexfloating],
        y: np.ndarray[np.complexfloating],
        params: np.ndarray[np.complexfloating],
    ):
        """
        Initialize the ConformalWelding class.

        Parameters:
        x (np.ndarray): Complex ndarray of shape (n,) representing initial x points.
        y (np.ndarray): Complex ndarray of shape (n,) representing initial y points.
        params (np.ndarray): Complex ndarray of shape (n+1,) representing parameters.
        """
        assert (
            x.ndim == y.ndim == params.ndim == 1
            and np.issubdtype(x.dtype, np.complexfloating)
            and np.issubdtype(y.dtype, np.complexfloating)
            and np.issubdtype(params.dtype, np.complexfloating)
        ), "Input must be 1D complex array"
        assert len(x) == len(y) == len(params) - 1, (
            "Length of x, y, params must be n, n, n+1"
        )

        self.params = params
        self._set_init_x(x)
        self._set_init_y(y)

    def _set_init_x(self, x: np.ndarray[np.complexfloating]):
        if x.size < 2 or not np.all(np.isfinite(x)):
            raise ConformalWeldingNumericalError("zipper produced an invalid outer boundary")
        x_angle = np.angle(x)
        x_angle_diff = np.diff(x_angle)
        min_pos = np.argmin(x_angle_diff)
        if x_angle_diff[min_pos] < 0:
            x_angle[min_pos + 1 :] += 2 * np.pi
        x = np.exp(x_angle * 1j)

        self.init_x_angle = x_angle
        self.init_x = x
        self.x = x

    def _set_init_y(self, y: np.ndarray[np.complexfloating]):
        self.init_y = y
        self.y = y

    @classmethod
    def from_angular(cls, welding: dict) -> "ConformalWelding":
        """从角速度引擎的焊接种构造（没有 zipper 的 slit 参数）。

        `welding` 由 `hbs.angular.angular_welding` 给出，字段为：

        - `init_x_angle` / `init_x`：源圆角度与点；
        - `init_y`：目标圆上的点（`|y| = 1`），随源角严格单调、绕圆一圈；
        - `parameter` / `boundary`：对应的形状参数与边界点（诊断用）。

        与 zipper 构造的对象**语义一致**（x、y 都在单位圆上，`rotate_x` 同样移动
        源角），区别是 `params is None`：`x_post_norm` / `y_post_norm` / `uniquify`
        需要 zipper 的 slit 参数，调用会抛 `NotImplementedError`。
        """
        self = cls.__new__(cls)
        x = np.asarray(welding["init_x"], complex)
        y = np.asarray(welding["init_y"], complex)
        if x.shape != y.shape or x.ndim != 1 or x.size < 2:
            raise ConformalWeldingNumericalError(
                "angular welding arrays must be 1D, equally sized, and non-degenerate"
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ConformalWeldingNumericalError("angular welding produced non-finite values")
        self.params = None
        self.init_x_angle = np.mod(np.asarray(welding["init_x_angle"], float), 2 * np.pi)
        self.init_x = np.exp(1j * self.init_x_angle)
        self.x = self.init_x
        self.init_y = y
        self.y = y
        self.parameter = np.asarray(welding["parameter"], float)
        self.boundary = np.asarray(welding["boundary"], complex)
        return self

    def _require_zipper_params(self, what: str) -> None:
        if getattr(self, "params", None) is None:
            raise NotImplementedError(
                f"{what} requires the zipper slit parameters, which the angular engine "
                "does not produce. Use get_conformal_welding(bound, method='zipper') "
                "if you need this step."
            )

    def uniquify(self):
        self._require_zipper_params("uniquify()")
        x_angle, idx = np.unique(np.angle(self.init_x), return_index=True)
        x = np.exp(x_angle * 1j)
        y = self.init_y[idx]
        y = np.exp(np.angle(y) * 1j)

        self._set_init_x(x)
        self._set_init_y(y)

    def linear_interp(self, num: int):
        """重采样到 `num` 个**均匀源角**，并保持 `x ↔ y` 配对。

        输出约定：`init_x_angle = 2πk/num`，`init_y[k]` 是该源角对应的目标圆点。

        两个不变量必须同时成立，否则下游会把两条焊接的 `y` 当成同一参数化：

        `self.x` 被写到从 0 起的均匀网格上；`init_x_angle` 保留插值前的原始起点，
        供 `rotate_x`（`init_x` 一路）与 `x_post_norm` 使用——`rotate_x` 转 `x` 时
        用的是**当前**网格，不是这张原始网格。`y` 的相位以首个点为基准归零——这是
        重建能落回原尺度与方位的前提，只消除**目标圆**的整体旋转。
        """
        x_angle_regular = np.arange(0, 2 * np.pi, 2 * np.pi / num)

        inter_func = interp1d(
            np.concatenate(
                [
                    self.init_x_angle - 2 * np.pi,
                    self.init_x_angle,
                    self.init_x_angle + 2 * np.pi,
                ]
            ),
            np.concatenate([self.init_y, self.init_y, self.init_y]),
            kind="linear",
            fill_value="extrapolate",  # 周期边界越界时外插（数值鲁棒）
        )
        y = inter_func(x_angle_regular)

        # `y` 的相位必须以首个点为基准归零：这是重建能落回原尺度/方位的前提
        # （去掉它，椭圆重建面积比从 1.00 变成 2.04）。归零只消除**目标圆**的
        # 整体旋转，不影响源圆参数化。
        y_angle = np.angle(y)
        self.y = np.exp((y_angle - y_angle[0]) * 1j)
        self.x = np.exp(x_angle_regular * 1j)

    def rotate_x(self, r):
        """把源圆参数化整体转 `r`：`x` / `init_x` 转，`y` 不动。

        转的是**当前 `x` 所在的网格**，不是 `init_x_angle` 那一张。两者在
        `linear_interp` 之后不再是同一张网格：`linear_interp` 把 `x` 写到从 0 起的
        均匀源角上（盘边界顶点就在这张网格上），而 `init_x_angle` 保留引擎的原始
        起点。角速度引擎的原始网格本来就是均匀的，两种写法等价；zipper 的不是
        （实测相邻源角相差 0.021–0.052 rad），转原始网格会让 `x` 与盘边界顶点错开
        3.1 rad，而 `field` 的边界值仍按顶点顺序与 `y` 配对。

        `y` 不动是对的：规范转的是**源圆**——`disk` 顶点随 `x` 一起转，目标圆的值
        不变。`field[:circle_num]` 与 `y` 逐位相等（`tests/test_pipeline_contract.py`
        钉住）就是这条不变量的体现。
        """
        x = np.exp(1j * np.mod(np.angle(self.x) - r, 2 * np.pi))
        # 原始网格的起点跟着转，供之后的 `linear_interp` / `x_post_norm` 使用。
        self._set_init_x(np.exp(1j * np.mod(self.init_x_angle - r, 2 * np.pi)))
        self.x = x

    def x_post_norm(self):
        """
        对外部点进行后归一化
        :param z: 输入点
        :param params: 参数
        :return: 归一化后的点，a 和 theta 参数
        """
        self._require_zipper_params("x_post_norm()")
        zinf = zipper_params([np.inf], self.params)
        a = -1 / np.conj(zinf)
        theta = 0
        x = mobius_inv(self.x, a, theta)
        x = np.flipud(x)
        self._set_init_x(x)

    def y_post_norm(self, eps=1e-8):
        """
        对内部点进行后归一化
        :param z: 输入点
        :return: 归一化后的点，a 和 theta 参数
        """
        self._require_zipper_params("y_post_norm()")
        p = np.append(self.y, [0, 1])
        for _ in range(200):
            center = np.mean(p[:-2])
            if np.abs(center) <= np.finfo(float).eps:
                break
            p = mobius(p, center)
        else:
            raise RuntimeError("y_post_norm Möbius centering did not converge")

        y = p[:-2]
        y_angle = np.angle(y)
        y_angle_diff = np.diff(np.insert(y_angle, -1, y_angle[0]))
        y_angle_diff[y_angle_diff < -np.pi] += 2 * np.pi
        y_angle_diff[(-0.1 < y_angle_diff) & (y_angle_diff < 0)] *= -1
        y_angle_diff[y_angle_diff < eps] = eps

        if y_angle_diff.sum() > 2 * np.pi:
            y_angle_diff *= 2 * np.pi / y_angle_diff.sum()

        y_angle_new = np.cumsum(y_angle_diff)
        y_angle_new = np.insert(y_angle_new[:-1] + y_angle[0], 0, y_angle[0])
        y_new = np.exp(y_angle_new * 1j)
        self._set_init_y(y_new)

    def get_y_angle_diff(self):
        y_angle = np.angle(self.y)
        y_angle = np.insert(y_angle, -1, y_angle[0])
        y_angle_diff = np.diff(y_angle)

        y_angle_diff[y_angle_diff < -np.pi] += 2 * np.pi
        y_angle_diff[(-0.1 < y_angle_diff) & (y_angle_diff < 0)] *= -1
        y_angle_diff[y_angle_diff == 0] = y_angle_diff[y_angle_diff > 0].min()

        if np.any(y_angle_diff < 0):
            raise ValueError("y_angle_diff must be non-negative")

        if y_angle_diff.sum() > 2 * np.pi:
            y_angle_diff *= 2 * np.pi / y_angle_diff.sum()

        return y_angle_diff

    def get_y_angle(self):
        y_angle_diff = self.get_y_angle_diff()
        y_angle = np.cumsum(y_angle_diff)[:-1]
        y_angle = np.insert(y_angle, 0, 0)
        return y_angle


def get_conformal_welding(
    bound: np.ndarray[np.floating],
    method: str = "angular",
    normalize: str = "default",
    resample_to: int = 500,
) -> ConformalWelding:
    """共形焊接：边界 → 源圆/目标圆上的对应点。

    与 `get_hbs` 走**同一段代码**（`hbs._pipeline` + `hbs._engines`），因此对同一
    形状、同一 `method`、同一 `normalize` 给出与 `get_hbs` 内部完全一致的 `(x, y)`。

    :param `bound`: `n x 2` 闭合边界点（任意绕向）
    :param `method`: `"angular"`（默认）或 `"zipper"`，与 `get_hbs` 同义；
        两个引擎互不替补，任一失败即抛错
    :param `normalize`: `"default"`（默认）/`"i2"`/`"m34"`/`"c0"`/`"none"`，与 `get_hbs` 同义。
        规范需要先算一次 μ 才能确定相位（焊接已定，不重解），效果与 `get_hbs`
        一致：只移源角，不动 `y`
    :param `resample_to`: 源圆采样点数，同时是弧长重采样的目标点数
    :return: `ConformalWelding`；`params is None`（没有 zipper 的 slit 参数）
    :raises RuntimeError: 该引擎在全部输入阶梯上失败
    """
    from hbs._engines import ENGINES
    from hbs._pipeline import NORMALIZATIONS, prepare_boundary, resolve_disk
    from hbs import gauges
    from hbs.extension import fem_extend
    from hbs.qc import get_beltrami_coefficient

    if method not in ENGINES:
        raise ValueError(f"method must be one of {sorted(ENGINES)}, got {method!r}")
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}")

    # 与 `get_hbs` 保持同一份预处理：角速度引擎不重采样，zipper 重采样。
    prepared = prepare_boundary(
        np.asarray(bound, float), resample_to, resample=(method != "angular")
    )
    # 引擎交回的焊接已经在源圆均匀网格上（`WeldingResult.values`），这里不再
    # 重采样——否则 `normalize="none"` 下会与 `get_hbs` 交回不同状态的焊接。
    welding = ENGINES[method](prepared, resample_to).welding

    if normalize != "none":
        disk, fem = resolve_disk(None, 0.01, resample_to)
        field, _ = fem_extend(disk, np.asarray(welding.y, complex), fem=fem)
        choice = gauges.select_gauge(
            get_beltrami_coefficient(field, disk), disk, normalize
        )
        if choice.phase is not None:
            welding.rotate_x(choice.phase)

    return welding


def zipper_welding_prepared(bound: np.ndarray[np.floating]) -> ConformalWelding:
    """对**已完成预处理**的边界跑 zipper 焊接（内部入口，不重复预处理）。

    调用方须先用 `hbs._pipeline.prepare_boundary` 规范化；重复预处理会再插值一次
    弧长网格，逐点偏移虽小但没必要。
    """
    bound = to_complex(np.asarray(bound, float))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x, _, x_params = zipper(bound)
            y, _, _ = zipper(np.flipud(bound))
            if min(x.size, y.size, x_params.size) < 2 or not (
                np.all(np.isfinite(x))
                and np.all(np.isfinite(y))
                and np.all(np.isfinite(x_params))
            ):
                raise ConformalWeldingNumericalError("zipper produced non-finite values")

            cw = ConformalWelding(x, y, x_params)
            cw.x_post_norm()
            cw.y_post_norm()
            cw.uniquify()
    except (ConformalWeldingNumericalError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ConformalWeldingNumericalError):
            raise
        raise ConformalWeldingNumericalError("zipper normalization failed") from exc
    return cw
