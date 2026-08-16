"""mu_chop 保梯度饱和——行为锁定测试。"""
import numpy as np

from hbs.utils.tool_functions import mu_chop


def test_all_values_below_one():
    # 任意 |mu|（含 >1）输出都必须 <1
    m = np.array([0.5, 0.9, 0.99999, 1.0, 1.3]) * np.exp(1j * 0.3)
    out = mu_chop(m)
    assert np.abs(out).max() < 1.0


def test_sub_threshold_unchanged():
    # |mu| < bound 应原样保留
    m = np.array([0.1, 0.5, 0.9]) * np.exp(1j * 0.5)
    out = mu_chop(m)
    assert np.allclose(out, m)


def test_saturation_preserves_order():
    # 顶部梯度保序：更大的 |mu| 输入 → 更大的 |mu| 输出（硬墙的键差异）
    m = np.array([0.9999, 0.99995, 1.0])
    out = np.abs(mu_chop(m))
    assert out[0] < out[1] < out[2]
    # 且顶端不再被抹平成同一个值
    assert not np.allclose(out[0], out[1])


def test_phase_preserved():
    m = np.array([1.0 + 1e-9j, 0.99999 * np.exp(1j * 1.1)])
    out = mu_chop(m)
    assert np.allclose(np.angle(out), np.angle(m), atol=1e-9)


def test_constant_old_api_backward_compat():
    # 旧接口 constant= 仍走硬墙行为（向后兼容红线）
    m = np.array([0.9999, 0.99995, 1.0]) * np.exp(1j * 0.4)
    out = mu_chop(m, constant=0.9999)
    assert np.allclose(np.abs(out), 0.9999)
    assert np.allclose(np.angle(out), np.angle(m))


def test_empty_noop():
    m = np.array([])
    out = mu_chop(m)
    assert out.shape == (0,)
