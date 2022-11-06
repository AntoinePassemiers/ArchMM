# -*- coding: utf-8 -*-
# setup.py
# Inspired by the sklearn package

import sys
import os
import platform
import shutil

import setuptools  # noqa
from distutils.command.clean import clean as Clean  # noqa
from distutils.command.sdist import sdist  # noqa

import traceback
import importlib
import builtins


builtins.__CRYPTOBOT_SETUP__ = True

from pkg_resources import parse_version

VERSION = '1.0.0'
DISTNAME = 'archmm'
DESCRIPTION = 'TODO'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Antoine Passemiers'
MAINTAINER_EMAIL = 'antoine.passemiers@gmail.com'

# For some commands, use setuptools
SETUPTOOLS_COMMANDS = {
    "develop",
    "release",
    "bdist_egg",
    "bdist_rpm",
    "bdist_wininst",
    "install_egg_info",
    "build_sphinx",
    "egg_info",
    "easy_install",
    "upload",
    "bdist_wheel",
    "--single-version-externally-managed",
}
extra_setuptools_args = {}


# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    description = 'Remove build artifacts from the source tree'

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("sklearn"):
            for filename in filenames:
                if any(
                    filename.endswith(suffix)
                    for suffix in (".so", ".pyd", ".dll", ".pyc")
                ):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in [".c", ".cpp"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {"clean": CleanCommand, "sdist": sdist}

# Custom build_ext command to set OpenMP compile flags depending on os and
# compiler. Also makes it possible to set the parallelism level via
# and environment variable (useful for the wheel building CI).
# build_ext has to be imported after setuptools
try:
    from numpy.distutils.command.build_ext import build_ext  # noqa

    class build_ext_subclass(build_ext):
        def finalize_options(self):
            super().finalize_options()
            if self.parallel is None:
                # Do not override self.parallel if already defined by
                # command-line flag (--parallel or -j)

                parallel = os.environ.get("SKLEARN_BUILD_PARALLEL")
                if parallel:
                    self.parallel = int(parallel)
            if self.parallel:
                print("setting parallel=%d " % self.parallel)

        def build_extensions(self):
            from sklearn._build_utils.openmp_helpers import get_openmp_flag

            #openmp_flag = get_openmp_flag(self.compiler)
            #for e in self.extensions:
            #    e.extra_compile_args += openmp_flag
            #    e.extra_link_args += openmp_flag

            build_ext.build_extensions(self)

    cmdclass['build_ext'] = build_ext_subclass

except ImportError:
    # Numpy should not be a dependency just to be able to introspect
    # that python 3.8 is required.
    pass


def configuration(parent_package="", top_path=None):
    if os.path.exists("MANIFEST"):
        os.remove("MANIFEST")

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid useless msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage(DISTNAME)

    return config


def check_package_status(package, min_version):
    """
    Returns a dictionary containing a boolean specifying whether given package
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = parse_version(package_version) >= parse_version(
            min_version
        )
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = "scikit-learn requires {} >= {}.\n".format(package, min_version)

    if package_status["up_to_date"] is False:
        if package_status["version"]:
            raise ImportError(
                "Your installation of {} {} is out-of-date.\n{}".format(
                    package, package_status["version"], req_str
                )
            )
        else:
            raise ImportError(
                "{} is not installed.\n{}".format(package, req_str)
            )


def setup_package():

    # TODO: Require Python 3.8 for PyPy when PyPy3.8 is ready
    # https://github.com/conda-forge/conda-forge-pinning-feedstock/issues/2089
    if platform.python_implementation() == "PyPy":
        python_requires = ">=3.7"
        required_python_version = (3, 7)
    else:
        python_requires = ">=3.8"
        required_python_version = (3, 8)

    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        cmdclass=cmdclass,
        python_requires=python_requires,
        package_data={"": ["*.pxd"]},
        **extra_setuptools_args,
    )

    commands = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if all(
        command in ("egg_info", "dist_info", "clean", "check") for command in commands
    ):
        # These actions are required to succeed without Numpy for example when
        # pip is used to install Scikit-learn when Numpy is not yet present in
        # the system.

        # These commands use setup from setuptools
        from setuptools import setup

        metadata["version"] = VERSION
        metadata["packages"] = ["sklearn"]
    else:
        if sys.version_info < required_python_version:
            required_version = "%d.%d" % required_python_version
            raise RuntimeError(
                "Scikit-learn requires Python %s or later. The current"
                " Python version is %s installed in %s."
                % (required_version, platform.python_version(), sys.executable)
            )

        # These commands require the setup from numpy.distutils because they
        # may use numpy.distutils compiler classes.
        from numpy.distutils.core import setup

        # Monkeypatches CCompiler.spawn to prevent random wheel build errors on Windows
        # The build errors on Windows was because msvccompiler spawn was not threadsafe
        # This fixed can be removed when we build with numpy >= 1.22.2 on Windows.
        # https://github.com/pypa/distutils/issues/5
        # https://github.com/scikit-learn/scikit-learn/issues/22310
        # https://github.com/numpy/numpy/pull/20640
        from numpy.distutils.ccompiler import replace_method
        from distutils.ccompiler import CCompiler
        #from sklearn.externals._numpy_compiler_patch import CCompiler_spawn
        from numpy.distutils.ccompiler import CCompiler_spawn

        replace_method(CCompiler, "spawn", CCompiler_spawn)

        metadata["configuration"] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
