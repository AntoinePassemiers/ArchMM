# -*- coding: utf-8 -*-

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy, os

source_folder = "Source"
source_files = [
    "Math.pyx",
    "KMeans.pyx",
    "ChangePointDetection.pyx",
    "Artifacts.pyx",
    "Fuzzy.pyx",
    "Parallel.pyx",
    "Queue.pyx",
    "IOHMM.pyx",
    "HMM.pyx",
    "HMM_Core.pyx",
    "DecisionTrees/Tree.pyx",
    "DecisionTrees/ID3.pyx"
]
source_filepaths = [os.path.join(source_folder, file) for file in source_files]

setup(
    ext_modules = cythonize(
        source_filepaths,
        language="c"),
    include_dirs = [numpy.get_include()]
)