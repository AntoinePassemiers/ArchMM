# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os

try:
    import Cython
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

USE_CYTHON = False # TO REMOVE

source_folder = "Source"
source_files = [
    "Math",
    "KMeans",
    "ChangePointDetection",
    "Artifacts",
    "Fuzzy",
    "Parallel",
    "Queue",
    "IOHMM",
    "HMM",
    "HMM_Core",
    "DecisionTrees/Tree",
    "DecisionTrees/ID3"
]

ext = '.pyx' if USE_CYTHON else '.c'
source_filepaths = [os.path.join(source_folder, file + ext) for file in source_files]

extensions = [Extension("ArchMM", source_filepaths, language = "c")]
if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(
        source_filepaths, 
        language = "c",
        # compiler_directives = {"embedsignature" : True}
    )

setup(
    name = "ArchMM",
    version = "1.0.0",
    description = "Machine Learning library with embedded HMM-based algorithms",
    author = "Antoine Passemiers",
    ext_modules = extensions,
    include_dirs = [np.get_include()]
)