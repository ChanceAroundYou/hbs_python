from setuptools import find_packages, setup

setup(
    name="hbs",
    version="2.3.3",
    author="Lin Chenran",
    author_email="chenranlin.17@gmail.com",
    description="Python library for computing Harmonic Beltrami Signature(HBS)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ChanceAroundYou/hbs_python",
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages(),  # 顶层只收 hbs/；GPL 隔离区在 package_dir 外
    install_requires=[
        "numpy>=1.20",  # numpy.typing.NDArray（hbs/_angular/direct.py）
        "scipy>=1.5",
    ],
    extras_require={
        "plot": ["matplotlib"],         # hbs.plotting 绘图
        "boundary": ["opencv-python"],  # hbs.utils.boundary 图像边界提取
        "all": ["matplotlib", "opencv-python"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
)
