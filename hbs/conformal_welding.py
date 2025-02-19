import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from .utils.mobius import mobius, mobius_inv
from .utils.tool_functions import to_complex
from .utils.zipper import zipper, zipper_params


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
        self.init_x = x
        self.init_x_angle = np.angle(x)
        self.x = x

    def _set_init_y(self, y: np.ndarray[np.complexfloating]):
        self.init_y = y
        self.y = y

    def uniquify(self):
        x_angle, idx = np.unique(np.angle(self.init_x), return_index=True)
        x = np.exp(x_angle * 1j)
        y = self.init_y[idx]
        y = np.exp(np.angle(y) * 1j)

        self._set_init_x(x)
        self._set_init_y(y)

    def linear_interp(self, num: int):
        x_angle_regular = np.arange(0, 2 * np.pi, 2 * np.pi / num)
        self.x = np.exp(x_angle_regular * 1j)

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
        )
        y = inter_func(x_angle_regular)
        self.y = np.exp(np.angle(y / y[0]) * 1j)

    def rotate_x(self, r):
        x_angle = np.mod(self.init_x_angle - r, 2 * np.pi)
        x = np.exp(x_angle * 1j)
        self._set_init_x(x)

    def x_post_norm(self):
        """
        对外部点进行后归一化
        :param z: 输入点
        :param params: 参数
        :return: 归一化后的点，a 和 theta 参数
        """
        zinf = zipper_params([np.inf], self.params)
        a = -1 / np.conj(zinf)
        theta = 0
        x = mobius_inv(self.x, a, theta)
        x = np.flipud(x)
        self._set_init_x(x)

    def y_post_norm(self):
        """
        对内部点进行后归一化
        :param z: 输入点
        :return: 归一化后的点，a 和 theta 参数
        """
        p = np.append(self.y, [0, 1])
        while True:
            center = np.mean(p[:-2])
            if np.abs(center) <= np.finfo(float).eps:
                break
            p = mobius(p, center)

        y = p[:-2]
        # y = np.flipud(y)
        self._set_init_y(y)

    def plot_x(self):
        plt.gca().set_aspect("equal", adjustable="box")
        plt.scatter(self.x.real, self.x.imag)
        plt.show()

    def plot_y(self):
        plt.gca().set_aspect("equal", adjustable="box")
        plt.scatter(self.y.real, self.y.imag)
        plt.show()

    def plot(self, is_interp=True):
        plt.gca().set_aspect("equal", adjustable="box")
        x_angle = np.angle(self.x)
        y_angle = np.angle(self.y)
        x_angle = np.mod(x_angle, 2 * np.pi)
        y_angle = np.mod(y_angle, 2 * np.pi)
        if is_interp:
            plt.plot(x_angle, y_angle, linestyle="-", linewidth=2)
        else:
            plt.scatter(x_angle, y_angle, s=2)
        plt.show()


def get_conformal_welding(bound: np.ndarray[np.floating]) -> ConformalWelding:
    assert (
        bound.ndim == 2
        and bound.shape[1] == 2
        and np.issubdtype(bound.dtype, np.floating)
    ), "bound must be n x 2 real array with float type"

    bound = to_complex(bound)
    x, _, x_params = zipper(bound)
    # x = np.flipud(x)
    y, _, _ = zipper(np.flipud(bound))

    cw = ConformalWelding(x, y, x_params)
    cw.x_post_norm()
    cw.y_post_norm()
    cw.uniquify()
    return cw
