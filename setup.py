from distutils.version import LooseVersion
import setuptools
from setuptools import setup

exec(open("admix/version.py").read())

setup(
    name="admix-kit",
    version=__version__,
    description="Tool kits for analyzing genetics data of admixed population",
    url="https://github.com/kangchenghou/admix-kit",
    author="Kangcheng Hou",
    author_email="kangchenghou@gmail.com",
    packages=["admix"],
    setup_requires=["numpy>=1.10"],
    python_requires=">=3.7",
    install_requires=[
        "Cython",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "dask[array]>=2021.11.2",
        "tqdm",
        "xarray",
        "statsmodels",
        "structlog",
        "fire",
        "pylampld @ git+https://github.com/bogdanlab/lamp-ld.git",
        "tinygwas @ git+https://github.com/bogdanlab/tinygwas.git",
        "dask-pgen @ git+https://github.com/KangchengHou/dask-pgen.git",
    ],
    entry_points={"console_scripts": ["admix=admix.cli:cli"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
)
