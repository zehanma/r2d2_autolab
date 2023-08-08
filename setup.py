"""
setup.py
PEP 621 switches most of Packaging to `pyproject.toml` -- yet keep a "dummy" setup.py for external code that has not
yet upgraded.
"""
from setuptools import setup, find_packages

# setup()


import os
import sys

root_dir = os.path.dirname(os.path.realpath(__file__))
setup(name='r2d2', package_dir = {'':'.'}, packages=find_packages())