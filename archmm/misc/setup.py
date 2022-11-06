# -*- coding: utf-8 -*-
# setup.py
# author: Antoine Passemiers

import os

from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('misc', parent_package, top_path)

    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
