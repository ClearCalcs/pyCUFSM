from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize
import numpy as np


class build_py(_build_py):

    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        if self.distribution.ext_modules == None:
            self.distribution.ext_modules = []

        cythonize(["pycufsm/analysis_c.pyx"])

        self.distribution.ext_modules.append(
            Extension(
                "pycufsm",
                sources=["pycufsm/analysis_c.c"],
                include_dirs=[np.get_include()],
            )
        )
