from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules = cythonize(
        [
            "Cyfiles/Math.pyx",
            "Cyfiles/KMeans.pyx",
            "Cyfiles/ChangePointDetection.pyx",
            "Cyfiles/Artifacts.pyx",
            "Cyfiles/Parallel.pyx",
            "Cyfiles/Queue.pyx",
            "Cyfiles/IOHMM.pyx",
            "Cyfiles/HMM.pyx",
            "Cyfiles/HMM_Core.pyx",
            "Cyfiles/Structs.pyx"
        ],
        language="c"),
    include_dirs = [numpy.get_include()]
)