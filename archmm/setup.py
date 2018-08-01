# -*- coding: utf-8 -*-

import os
import sys
from distutils.extension import Extension

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup


source_folder = "archmm"
sub_packages = [
    "ann",
    "estimation",
    "misc",
]
source_files = [
    (["hmm.c"], "hmm"),
    (["iohmm.c"], "iohmm"),
    (["mrf.c"], "mrf"),
    (["math.c"], "math"),
    (["stats.c"], "stats"),
    (["ann/layers.py"], "ann.layers"),
    (["ann/subnetworks.py"], "ann.subnetworks"),
    (["estimation/clustering.c"], "estimation.clustering"),
    (["estimation/cpd.c", "estimation/kernel_.c"], "estimation.cpd"),
    (["estimation/queue.c"], "estimation.queue"),
    (["misc/check_data.py"], "check_data"),
    (["misc/exceptions.py"], "exceptions"),
    (["misc/topology.py"], "topology"),
    (["misc/utils.py"], "utils")
]

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
        sources=sources,
        include_dirs=include_dirs+[os.curdir],
        libraries=libraries,
        extra_compile_args=extra_compile_args)

np_setup(**config.todict())
