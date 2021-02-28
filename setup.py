from distutils.version import LooseVersion
from io import open

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Added support for environment markers in install_requires.
if LooseVersion(setuptools.__version__) < "36.2":
    raise ImportError("setuptools>=36.2 is required")


class build_ext(build_ext):
    def finalize_options(self):
        # The key point: here, Cython and numpy will have been installed by
        # pip.
        from Cython.Build import cythonize
        import numpy as np
        import numpy.distutils

        self.distribution.ext_modules[:] = cythonize("**/*.pyx")
        # Sadly, this part needs to be done manually.
        for ext in self.distribution.ext_modules:
            for k, v in np.distutils.misc_util.get_info("npymath").items():
                setattr(ext, k, v)
            ext.include_dirs = [np.get_include()]

        super().finalize_options()

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        super().build_extensions()


setup(
    name="admix-tools",
    version="0.1",
    description="Toolbox for analyzing genetics data from admixed population",
    url="https://github.com/kangchenghou/admix-tools",
    author="Kangcheng Hou",
    author_email="kangchenghou@gmail.com",
    packages=["admix"],
    setup_requires=["Cython", "numpy>=1.10"],
    cmdclass={"build_ext": build_ext},
    ext_modules=[Extension("", [])],
)
