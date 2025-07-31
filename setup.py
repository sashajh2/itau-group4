# I've never done this before, but it makes the project pip-installable as a package for use in notebooks/external scripts
from setuptools import setup, find_packages

setup(
    name="itau_project",
    version="0.1",
    packages=find_packages(),  # auto-detects folders with __init__.py
)
