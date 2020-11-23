#!/usr/bin/env python

import os.path
from setuptools import setup, find_packages

# Get __version__ from pyigtl/_version.py
base_dir = os.path.dirname(os.path.realpath(__file__))
version_file = os.path.join(base_dir, 'pyigtl', '_version.py')
print("base_dir="+base_dir)
print("version_file="+version_file)
with open(version_file) as f:
    exec(f.read())
# pylint/flake does not know that this script injects a variable, so we need to disable checks
VERSION = __version__  # pylint:disable=undefined-variable  # noqa: F821

# Get long description from README.md
base_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(base_path, 'README.md')) as f:
    LONG_DESCRIPTION = f.read() 

opts = dict(
    name="pyigtl",
    python_requires='>=3.6',
    version=VERSION,
    maintainer="Andras Lasso",
    maintainer_email="lasso@queensu.ca",
    author="Andras Lasso, Daniel Hoyer Iversen, Kyle Sunderland",
    author_email="lasso@queensu.ca",
    description="Python interface for OpenIGTLink",
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url="https://github.com/lassoan/pyigtl",
    download_url="https://github.com/lassoan/pyigtl/archive/master.zip",
    keywords="openigtlink igt medical imaging",
    classifiers=[
      "License :: OSI Approved :: MIT License",
      "Intended Audience :: Developers",
      "Intended Audience :: Healthcare Industry",
      "Intended Audience :: Science/Research",
      "Development Status :: 4 - Beta",
      "Programming Language :: Python :: 3",
      "Operating System :: OS Independent",
      "Topic :: Scientific/Engineering :: Medical Science Apps.",
      "Topic :: System :: Networking"
      ],
    packages=find_packages(),
    install_requires=['crcmod', 'numpy'],
)

if __name__ == '__main__':
    setup(**opts)
