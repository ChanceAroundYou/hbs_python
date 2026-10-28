"""打包自足性测试：wheel 必须能在没有研究工作区的前提下独立工作。

背景（2.0.0 之前的真实缺陷）：`hbs.angular` 曾用
`import direct_boundary_hbs` / `from joint_hbs_optimizer import ...` 引两个只存在于
研究仓库 `src/` 下的模块。源码树里因为这些模块恰好可导入而始终绿，装成 wheel
后默认路径直接 `ModuleNotFoundError`。本测试锁住三条：

1. 包内**不许**出现指向研究仓库顶层模块的 import；
2. 引擎确实随包 vendored（`hbs._angular.direct`）；
3. 在子进程里用 `sys.path` 屏蔽研究仓库后，默认路径仍能跑通。
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "hbs"

# 只存在于研究仓库 src/ 的顶层模块名；wheel 里不许出现对它们的引用
FOREIGN_MODULES = ("direct_boundary_hbs", "joint_hbs_optimizer", "upgraded_hbs")

_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+({m})\b|import\s+({m})\b)".format(m="|".join(FOREIGN_MODULES)),
    re.MULTILINE,
)


def _python_files():
    return sorted(PKG.rglob("*.py"))


def test_no_imports_from_research_repo():
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for match in _IMPORT_RE.finditer(text):
            offenders.append(f"{path.relative_to(PKG.parent)}: {match.group(0).strip()}")
    assert not offenders, "包内引用了研究仓库顶层模块：\n" + "\n".join(offenders)


@pytest.mark.parametrize("mod", ["hbs._angular.direct"])
def test_engine_is_vendored(mod):
    import importlib

    module = importlib.import_module(mod)
    assert Path(module.__file__).resolve().is_relative_to(PKG), (
        f"{mod} 解析到包外：{module.__file__}"
    )


def test_no_gpl_modules_in_package():
    """GPL 代码不得进入发布树：`hbs` 以 MIT 分发，混入 GPL 会改变分发条款。

    `angle_slit_welding.py` 是 Marshall zipper 程序的 GPL 逐字移植，已移到
    仓库根 `unpublished_gpl/`（非包目录，`find_packages()` 不收集、wheel 不含）。
    这里锁住它不会因为有人放回 `hbs/utils/` 而静默重新进入 wheel。
    """
    gpl_markers = ("GNU General Public License", "GPL")
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if path.name == "angle_slit_welding.py" or any(m in text for m in gpl_markers):
            offenders.append(str(path.relative_to(PKG.parent)))
    assert not offenders, (
        "包内出现 GPL 代码；请移到 unpublished_gpl/（MIT 与 GPL 不可合并分发）：\n"
        + "\n".join(offenders)
    )


def test_import_emits_no_warnings():
    """`import hbs` 必须零警告：下游 `pytest -W error` / CI 会被警告直接打挂。

    2.0.0 之前 `hbs.py` 顶层 import 了一个在模块级 `warnings.warn` 的实验模块，
    导致每次 import 都发 DeprecationWarning。用子进程验证，避免受本进程
    warning filter 影响。
    """
    script = "import hbs; print('OK')"
    proc = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
        env={"PYTHONPATH": str(PKG.parent), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, (
        "`import hbs` 在 -W error 下失败（有警告被升级为错误）：\n" + proc.stderr
    )


def test_default_path_works_with_research_repo_hidden():
    """子进程里只给出本包路径（屏蔽研究仓库）后，默认 angular 路径仍须端到端跑通。

    `PYTHONPATH` 只指向 `PKG.parent`，所以子进程里 `hbs` 解析到本仓库源码，而研究
    仓库的 `src/`（`direct_boundary_hbs` 等所在处）不在搜索路径上 —— 这正是
    「源码树里可导入、wheel 里断裂」那个缺陷的复现条件。真实 wheel 安装验证另做
    （见 CLAUDE.md 的打包自足性流程），本测试只负责 import 图。
    """
    script = textwrap.dedent(
        """
        import sys, warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        import hbs

        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        bound = np.stack([1.2 * np.cos(t), np.sin(t)], 1)
        mu, field, cw, disk = hbs.get_hbs(bound)
        assert cw is not None, "2.2.0：两条路径都带焊接（统一从角速度求）"
        assert mu.shape == (disk.face_num,)
        rec, *_ = hbs.reconstruct_from_hbs(mu, disk)
        assert rec.shape == (disk.circle_num, 2)
        assert np.all(np.isfinite(rec))

        leaked = [m for m in sys.modules if m in %r]
        assert not leaked, f"研究仓库模块被导入：{leaked}"
        print("SMOKE_OK")
        """
        % (FOREIGN_MODULES,)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent),  # 不在仓库根，避免相对导入兜底
        env={
            "PYTHONPATH": str(PKG.parent),  # 只有本包；研究仓库不在路径上
            "PATH": "/usr/bin:/bin",
        },
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "SMOKE_OK" in proc.stdout


def test_no_annotations_evaluate_at_import_on_py38():
    """`python_requires=">=3.8"` 必须被兑现：注解不能在 import 期被求值。

    没有 `from __future__ import annotations` 时，三种注解会在 3.8/3.9 上炸：

    - `X | None`（PEP 604）→ `TypeError: unsupported operand type(s) for |`
    - `np.ndarray[np.floating]` → `TypeError: Type subscription requires
      python >= 3.9`（numpy 1.x 的 `ndarray` 在 3.8 不可下标）
    - `list[int]` / `dict[str, int]` / `tuple[...]` → 同类 `TypeError`

    前两类都是已发布 1.1.0 的实际缺陷（3.8 上 `import hbs` 直接失败），
    第三类只要有人写就会重演。用 AST 判定，不看源码文本 —— docstring 里的
    `|mu| < bound` 不是注解。
    """
    import ast

    builtin_generics = {"list", "dict", "tuple", "set", "frozenset", "type"}

    def _offending_annotation(ann) -> bool:
        for sub in ast.walk(ann):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
                return True
            if isinstance(sub, ast.Subscript):
                head = sub.value
                if isinstance(head, ast.Name) and head.id in builtin_generics:
                    return True
                if isinstance(head, ast.Attribute) and head.attr == "ndarray":
                    return True
        return False

    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if "from __future__ import annotations" in text:
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            # 函数参数/返回值注解，以及模块/类级别变量注解
            annotations = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for arg in (
                    *args.posonlyargs,
                    *args.args,
                    *args.kwonlyargs,
                    args.vararg,
                    args.kwarg,
                ):
                    if arg is not None and arg.annotation is not None:
                        annotations.append(arg.annotation)
                if node.returns is not None:
                    annotations.append(node.returns)
            elif isinstance(node, ast.AnnAssign):
                annotations.append(node.annotation)
            for ann in annotations:
                if _offending_annotation(ann):
                    offenders.append(f"{path.relative_to(PKG.parent)}:{ann.lineno}")
    assert not offenders, (
        "这些注解会在 3.8/3.9 的 import 期求值失败；"
        "加 `from __future__ import annotations`：\n" + "\n".join(sorted(set(offenders)))
    )
