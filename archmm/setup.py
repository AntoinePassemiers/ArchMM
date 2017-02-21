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
    "estimation",
    "plot",
    "svm",
    "tests",
    "trees"
]
source_files = [
    (["artifacts.c"], "artifacts"),
    (["core.py"], "core"),
    (["format_data.py"], "format_data"),
    (["fuzzy.c"], "fuzzy"),
    (["hmm.c"], "hmm"),
    (["iohmm.c"], "iohmm"),
    (["math.c"], "math"),
    (["parallel.c"], "parallel"),
    (["queue.c"], "queue"),
    (["structs.c"], "structs"),
    (["utils.c"], "utils"),
    (["adaptation/mllr.py"], "adaptation.mllr"),
    (["ann/cnn.py"], "ann.cnn"),
    (["ann/layers.py"], "ann.layers"),
    (["ann/mlp.py"], "ann.mlp"),
    (["estimation/clustering.c"], "estimation.clustering"),
    (["estimation/cpd.c"], "estimation.cpd"),
    (["plot/tools.py"], "plot.tools"),
    (["svm/kernel.c", "svm/kernel_.c"], "svm.kernel"),
    (["svm/svm.py"], "svm.svm"),
    (["tests/HMM_test.py"], "tests.HMM_tests"),
    (["tests/test_clustering.py"], "tests.test_clustering"),
    (["tests/test_kernels.py"], "tests.test_kernels"),
    (["tests/test_trees.py"], "tests.test_trees"),
    (["trees/tree.c", "trees/id3_.c", "trees/id4_.c",
      "trees/queue_.c", "trees/utils_.c"], "trees.tree"),
    (["tests/utils.py"], "tests.test_utils")
]

extra_compile_args = [
    "-std=c99", 
    "-fno-strict-aliasing",
    "-D_FORTIFY_SOURCE=2",
    "-DNDEBUG",
    "-fwrapv",
    "-g",
    # "-fstack-protector-strong",
    "-ldl",
    "-lm",
    "-lpthread",
    "-lutil",
    "-O3",
    "-O0",
    "-Wall",
    "-Werror=format-security",
    "-Wstrict-prototypes",
    "-Wuninitialized"
]

libraries = ["m"] if os.name == "posix" else list()
include_dirs = [np.get_include()]

config = Configuration(source_folder, "", "")
for sub_package in sub_packages:
    config.add_subpackage(sub_package)
for sources, extension_name in source_files:
    sources = [os.path.join(source_folder, source) for source in sources]
    print(extension_name, sources)
    config.add_extension(
        extension_name, 
        sources = sources,
        include_dirs = include_dirs + [os.curdir],
        libraries = libraries,
        extra_compile_args = extra_compile_args
    )

np_setup(**config.todict())