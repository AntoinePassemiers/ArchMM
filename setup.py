# -*- coding: utf-8 -*-

import os, sys, subprocess
from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup
from numpy.distutils.numpy_distribution import NumpyDistribution

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    

source_folder = "archmm"
source_files = [
    "anomaly.pyx",
    "hmm.pyx",
    "iohmm.pyx",
    "mrf.pyx",
    "math.pyx",
    "stats.pyx",
    "utils.pyx",
    "estimation/clustering.pyx",
    "estimation/cpd.pyx",
    "estimation/queue.pyx"
]

def configuration(parent_package=str(), top_path=None):
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage("archmm")

    return config

setup_args = {
    "name" : "ArchMM",
    "version" : "1.0.0",
    "description" : "Machine Learning library with embedded HMM-based algorithms",
    "long_description" : str(), # TODO
    "author" : "Antoine Passemiers",
    "classifiers" : [ # TODO
    ],
    "configuration" : configuration
}

extensions = list()
for source_file in source_files:
    source_filepath = os.path.join(source_folder, source_file)
    sources = [source_filepath]
    extension_name = ".".join(["archmm", source_file]).replace("/", ".")
    extension_name = os.path.splitext(extension_name)[0]
    print(extension_name, sources)
    extensions.append(
        Extension(
            extension_name,
            sources,
            language="c",
            include_dirs=[np.get_include()]))

build_cmds = {'install', 'build', 'build_ext'}
GOT_BUILD_CMD = len(set(sys.argv) & build_cmds) != 0

if USE_CYTHON and GOT_BUILD_CMD:
    # Setting "archmm" as the root package
    # This is to prevent cython from generating inappropriate variable names
    # (because it is based on a relative path)
    init_path = os.path.join(os.path.realpath(__file__), "../__init__.py")
    if os.path.isfile(init_path):
        os.remove(init_path)
        print("__init__.py file removed")
    # Generates the C files, but does not compile them
    extensions = cythonize(extensions, language="c")

if GOT_BUILD_CMD:
    np_setup(**setup_args)