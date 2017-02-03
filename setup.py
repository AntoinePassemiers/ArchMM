# -*- coding: utf-8 -*-

import os, sys, subprocess
from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup

try:
    import Cython
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


source_folder = "archmm"
source_files = [
    "utils",
    "math",
    "structs",
    "fuzzy",
    "clustering",
    "cpd",
    "artifacts",
    "parallel",
    "queue",
    "iohmm",
    "hmm",
    "trees/tree",
    "trees/id3"
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

ext = '.pyx' if USE_CYTHON else '.c'
source_filepaths = [os.path.join(source_folder, file + ext) for file in source_files]

extensions = list()
for source_file in source_files:
    source_filepath = os.path.join(source_folder, source_file + ext)
    print(source_file, source_filepath)
    extensions.append(
        Extension(".".join(["archmm", source_file]), 
                  [source_filepath],
                  language = "c",
                  include_dirs = [np.get_include()]
        )
    )

if USE_CYTHON:
    # Generates the C files, but does not compile them
    from Cython.Build import cythonize
    extensions = cythonize(
        extensions,
        language = "c"
    )

np_setup(**setup_args)