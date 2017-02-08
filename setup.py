# -*- coding: utf-8 -*-

import os, sys, subprocess
from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    

source_folder = "archmm"
source_files = [
    "utils.pyx",
    "math.pyx",
    "structs.pyx",
    "fuzzy.pyx",
    "artifacts.pyx",
    "parallel.pyx",
    "queue.pyx",
    "iohmm.pyx",
    "hmm.pyx",
    "estimation/clustering.pyx",
    "estimation/cpd.pyx",
    "trees/tree.pyx"
]

def configuration(parent_package = str(), top_path = None):
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py = True,
                       assume_default_configuration = True,
                       delegate_options_to_subpackages = True,
                       quiet = True)
    config.add_subpackage("archmm")

    return config

setup_args = {
    "name" : "ArchMM",
    "version" : "1.0.0",
    "description" : "Machine Learning library with embedded HMM-based algorithms",
    "long_description" : str(), # TODO
    "author" : "Antoine Passemiers",
    "classifiers" : [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Communications :: Email',
        'Topic :: Software Development :: Bug Tracking',
    ],
    "configuration" : configuration
}

extensions = list()
for source_file in source_files:
    source_filepath = os.path.join(source_folder, source_file)
    sources = [source_filepath]
    print(source_file, sources)
    extensions.append(
        Extension(".".join(["archmm", source_file]),
                  sources,
                  language = "c",
                  include_dirs = [np.get_include()]
        )
    )

if USE_CYTHON:
    # Setting "archmm" as the root package
    # This is to prevent cython from generating inappropriate variable names
    # (because it is based on a relative path)
    init_path = os.path.join(os.path.realpath(__file__), "../__init__.py")
    if os.path.isfile(init_path):
        os.remove(init_path)
        print("__init__.py file removed")
    # Generates the C files, but does not compile them
    extensions = cythonize(
        extensions,
        language = "c"
    )

np_setup(**setup_args)