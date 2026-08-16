"""绘图解耦测试：核心 import 不得加载 matplotlib，绘图模块可独立 import。"""
import subprocess
import sys

import numpy as np

from hbs import get_hbs
from hbs.mesh import get_rect, get_unit_disk


def test_core_import_does_not_load_matplotlib():
    # 在干净解释器里 import hbs，确认 matplotlib 未被拉入（解耦地狱门）
    code = (
        "import sys; import hbs; import hbs.mesh; import hbs.conformal_welding;"
        "import hbs.qc.lsqc;"
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "核心 import 不应加载 matplotlib"


def test_plotting_module_importable_lazily():
    # hbs.plotting 可 import，且 import 本身也不加载 matplotlib（惰性）
    code = (
        "import sys; import hbs.plotting;"
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_plot_functions_exist():
    import matplotlib.pyplot as plt  # noqa: F401  # 显式拉入

    from hbs.plotting import (
        plot_boundary,
        plot_conformal,
        plot_mapping,
        plot_mesh,
        plot_mu,
        plot_welding,
    )

    assert callable(plot_mesh)
    assert callable(plot_mu)
    assert callable(plot_mapping)
    assert callable(plot_boundary)
    assert callable(plot_conformal)
    assert callable(plot_welding)
