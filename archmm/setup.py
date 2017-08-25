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
    "mms",
    "plot",
    "kernel",
    "tests"
]
source_files = [
    (["anomaly.c"], "anomaly"),
    (["chains.c"], "chains"),
    (["core.py"], "core"),
    (["format_data.py"], "format_data"),
    (["fuzzy.c"], "fuzzy"),
    (["dbn.c"], "dbn"),
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
    (["ann/optimizers.py"], "ann.optimizers"),
    (["estimation/clustering.c"], "estimation.clustering"),
    (["estimation/cpd.c", "estimation/kernel_.c"], "estimation.cpd"),
    (["plot/tools.py"], "plot.tools"),
    (["kernel/kernel.c"], "kernel.kernel"),
    (["kernel/svm.py"], "kernel.svm"),
    (["mms/msd.c"], "msd"),
    (["tests/HMM_test.py"], "tests.HMM_tests"),
    (["tests/test_clustering.py"], "tests.test_clustering"),
    (["tests/test_kernels.py"], "tests.test_kernels"),
    (["tests/utils.py"], "tests.test_utils")
]

"""
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
"""
extra_compile_args = list()


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