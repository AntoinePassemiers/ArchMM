from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules = cythonize(
        [
            "Source/Math.pyx",
            "Source/KMeans.pyx",
            "Source/ChangePointDetection.pyx",
            "Source/Artifacts.pyx",
            "Source/Parallel.pyx",
            "Source/Queue.pyx",
            "Source/IOHMM.pyx",
            "Source/HMM.pyx",
            "Source/HMM_Core.pyx",
            "Source/DecisionTrees/Tree.pyx",
            "Source/DecisionTrees/ID3.pyx"
        ],
        language="c"),
    include_dirs = [numpy.get_include()]
)