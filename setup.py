from distutils.version import LooseVersion
import setuptools
from setuptools import setup

# Added support for environment markers in install_requires.
if LooseVersion(setuptools.__version__) < "36.2":
    raise ImportError("setuptools>=36.2 is required")

setup(
    name="admix-kit",
    version="0.1",
    description="Tool kits for analyzing genetics data of admixed population",
    url="https://github.com/kangchenghou/admix-tools",
    author="Kangcheng Hou",
    author_email="kangchenghou@gmail.com",
    packages=["admix"],
    setup_requires=["numpy>=1.10"],
    entry_points={"console_scripts": ["admix=admix.cli:cli"]},
)
