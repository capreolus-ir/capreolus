import setuptools
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
import os


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="capreolus",
    version="0.1.3",
    author="Andrew Yates",
    author_email="capreolus@mpi-inf.mpg.de",
    description="A toolkit for end-to-end neural ad hoc retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://capreolus.ai",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch==1.2.0",
        "torchvision",
        "cython",
        "pre-commit",
        "PyYAML==5.1.1",
        "sacred==0.7.5",
        "SQLAlchemy==1.3.5",
        "psycopg2-binary==2.8.3",
        "matplotlib==3.1.0",
        "pytorch-transformers==1.1.0",
        "colorlog==4.0.2",
        "pytrec-eval==0.4",
        "pycapnp==0.6.4",
        "pytest",
        "pyjnius==1.2.1",
        "pescador==2.0.2",
        "pymagnitude==0.1.120",
        "h5py==2.9.0",
        "pytorch-pretrained-bert==0.4",
        "nltk==3.4.5",
        "pymongo==3.9.0",
        "lz4==2.1.10",
        "xxhash==1.3.0",
        "annoy==1.15.2",
        "fasteners==0.15",
        "django==2.2.9",
        "pytest-mock==1.10.4",
        "mock",
        "pyserini==0.7.0.0",
        "scipy==1.3.0",
    ],
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    python_requires=">=3.6",
    cmdclass={"develop": PostDevelopCommand, "install": PostInstallCommand},
    include_package_data=True,
    scripts=["scripts/capreolus"],
)
