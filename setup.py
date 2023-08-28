#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='AnalysisTools',
    version='0.1.0',
    packages=find_packages(include=['AnalysisTools', 'AnalysisTools.*'])
)