# -*- coding: utf-8 -*-

import os, sys
from distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup


source_folder = "archmm"
sub_packages = [
    "adaptation",
    "ann",
    "svm",
    "trees",
    "tests"
]

libraries = ["m"] if os.name == "posix" else list()
extra_compile_args = ["-O3"]
include_dirs = [np.get_include()]

config = Configuration(source_folder, "", "")
for sub_package in sub_packages:
    config.add_subpackage(sub_package)
for sub_package in sub_packages + ["."]:
    sub_package_path = os.path.join(source_folder, sub_package)
    for source_file in os.listdir(sub_package_path):
        basename, ext = os.path.splitext(source_file)
        to_add = False
        if ext == ".c":
            if not basename.endswith("_"):
                to_add = True
        elif ext == ".py":
            if basename not in ["setup", "__init__"]:
                to_add = True
        if to_add:
            if sub_package == ".":
                extension_name = basename
            else:
                extension_name = ".".join([sub_package, basename])
            print(extension_name, os.path.join(sub_package_path, source_file))
            config.add_extension(
                extension_name, 
                sources = [os.path.join(sub_package_path, source_file)],
                include_dirs = include_dirs + [os.curdir],
                libraries = libraries,
                extra_compile_args = extra_compile_args
            )

np_setup(**config.todict())