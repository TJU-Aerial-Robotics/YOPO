import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

setup(
    name='flightpolicy',
    version='0.0.1',
    author='Junjie Lu',
    author_email='lqzx1998@tju.edu.cn',
    description='A Learning-based Planner for Autonomous Navigation',
    long_description='',
    packages=['flightpolicy'],
)
