import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
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

# from https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "rt") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="capreolus",
    version=get_version("capreolus/__init__.py"),
    author="Andrew Yates",
    author_email="initial-then-last@mpi-inf.mpg.de",
    description="A toolkit for end-to-end neural ad hoc retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://capreolus.ai",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch==1.6",
        "torchvision==0.7",
        "cython",
        "pre-commit",
        "profane>=0.2.0",
        "SQLAlchemy",
        "sqlalchemy-utils",
        "psycopg2-binary",
        "matplotlib",
        "colorlog==4.0.2",
        "pytrec_eval>=0.5",
        "pytest",
        "pyjnius==1.2.1",
        "pymagnitude==0.1.143",
        "h5py",
        "nltk==3.4.5",
        "pytest-mock",
        "mock",
        "pyserini==0.9.3.0",
        "numpy",
        "scipy",
        "google-api-python-client",
        "oauth2client",
        "tensorflow==2.3.1",
        "transformers==3.1.0",
        "tensorflow-ranking",
        "Pillow",
        "beautifulsoup4",
        "lxml",
        "scispacy",
        "spacy",
        "pandas",
    ],
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    python_requires=">=3.6",
    cmdclass={"develop": PostDevelopCommand, "install": PostInstallCommand},
    include_package_data=True,
    scripts=["scripts/capreolus"],
)
