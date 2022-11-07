# -*- coding: utf-8 -*-
# setup.py
# author: Antoine Passemiers

import sys
import os

from numpy.distutils.core import setup
from sklearn._build_utils import cythonize_extensions


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('archmm', parent_package, top_path)

    config.add_subpackage('distributions')

    for filename in ['hmm']:
        config.add_extension(
            filename,
            sources=[f'{filename}.pyx'],
            include_dirs=[numpy.get_include()],
            libraries=libraries,
        )

    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
