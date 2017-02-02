# -*- coding: utf-8 -*-

import os, sys
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

source_folder = "Source"
source_files = [
    "utils",
    "math",
    "structs",
    "clustering",
    "cpd",
    "artifacts",
    "fuzzy",
    "parallel",
    "queue",
    "iohmm",
    "hmm",
    "trees/tree",
    "trees/id3"
]

sub_packages = [
    "adaptation",
    "ann",
    "svm",
    "trees",
    "tests"
]

setup_args = {
    "name" : "archmm",
    "version" : "1.0.0",
    "description" : "Machine Learning library with embedded HMM-based algorithms",
    "author" : "Antoine Passemiers",
    "classifiers" : [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Topic :: Communications :: Email',
        'Topic :: Software Development :: Bug Tracking',
    ],
}

libraries = ["m"] if os.name == "posix" else list()

include_dirs = [np.get_include()]

ext = '.pyx' if USE_CYTHON else '.c'
source_filepaths = [os.path.join(source_folder, file + ext) for file in source_files]

extensions = [Extension("archmm", source_filepaths, 
                        language = "c",
                        include_dirs = [np.get_include()])]

if USE_CYTHON:
    # Generates the C files, but does not compile them
    from Cython.Build import cythonize
    extensions = cythonize(
        extensions,
        language = "c"
    )

config = Configuration("archmm", "", "")
for sub_package in sub_packages:
    config.add_subpackage(sub_package, subpackage_path = source_folder)
for sub_package in sub_packages + [""]:
    sub_package_path = os.path.join(source_folder, sub_package)
    for source_file in os.listdir(sub_package_path):
        basename, ext = os.path.splitext(source_file)
        to_add = False
        if ext == ".c":
            to_add = True
        elif ext == ".py":
            if basename not in ["setup", "__init__"]:
                to_add = True
        if to_add:
            if sub_package == "":
                extension_name = basename
            else:
                extension_name = ".".join([sub_package, basename])
            config.add_extension(
                extension_name,
                sources = [os.path.join(sub_package_path, source_file)],
                include_dirs = include_dirs + [os.curdir],
                libraries = libraries,
            )

config.dict_append(**setup_args)
np_setup(**config.todict())