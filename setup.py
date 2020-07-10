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
    author="Andrew Yates, Kevin Martin Jose, Xinyu Zhang, Siddhant Arora, Wei Yang, Jimmy Lin",
    author_email="",
    description="A toolkit for end-to-end neural ad hoc retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://capreolus.ai",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch==1.4",
        "torchvision==0.5",
        "cython",
        "pre-commit",
        "profane>=0.1.6",
        "SQLAlchemy",
        "sqlalchemy-utils",
        "psycopg2-binary",
        "matplotlib",
        "pytorch-transformers==1.1.0",
        "colorlog==4.0.2",
        "pytrec_eval_git",
        "pytest",
        "pyjnius==1.2.1",
        "pymagnitude==0.1.120",
        "h5py",
        "pytorch-pretrained-bert==0.4",
        "nltk==3.4.5",
        "lz4==2.1.10",
        "xxhash==1.3.0",
        "annoy==1.15.2",
        "fasteners==0.15",
        # "django==2.2.13",
        "pytest-mock==1.10.4",
        "mock",
        "pyserini==0.9.3.0",
        "numpy",
        "scipy",
        "keras",
        "google-api-python-client",
        "oauth2client",
        "tensorflow==2.2",
        "transformers",
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
