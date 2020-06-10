# Installation

This section covers installing Capreolus via its pip package or from source.

### Java
Java 11 is required. On Debian-based distributions, this can be installed with `sudo apt install openjdk-11-jre`. You may additionally need to set the `JAVA_HOME` environment variable and/or use `update-alternatives` to ensure the correct version of Java is used by default.
```
$ java -version
openjdk version "11.0.5" 2019-10-15
OpenJDK Runtime Environment (build 11.0.5+10-post-Ubuntu-0ubuntu1.119.04)
OpenJDK 64-Bit Server VM (build 11.0.5+10-post-Ubuntu-0ubuntu1.119.04, mixed mode, sharing)
```

### Python
Setup a Python 3.7+ environment in your home directory. We recommend using Conda for performance reasons, but this is not strictly necessary.

 a) *Recommended conda approach*: install [pyenv](https://github.com/pyenv/pyenv) into your home directory, and then use pyenv to install a miniconda (or anaconda) distribution with Python 3.7+.
 
 b) Alternate conda approach: [install a miniconda distribution](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) with Python 3.7+.
 
 c) Alternate approach with system Python: install Python 3.7+ with your system's package manager (e.g., `sudo apt install python3`). If you do not create and activate a virtual environment (*venv*) as described below, you will need to use `sudo` when installing packages with `pip` below.
 
 You may optionally [setup a virtual environment using `venv`](https://docs.python.org/3/tutorial/venv.html) to isolate Capreolus and its dependencies from other packages. This is especially useful if using a system Python, because it allows you to install packages (for your own user) without `sudo`.
 
### Installing Capreolus
#### Via pip package (**recommended**)
`pip install capreolus`
 
#### From source
- Clone the repository: `git clone git@github.com:capreolus-ir/capreolus.git` and `cd` into it.
- Install [PyTorch 1.2](https://pytorch.org/get-started/previous-versions/#v120). Note that the installation command differs depending on your CUDA version and whether you're using a Conda distribution (see "Conda" section) or a  system Python (see "Wheel" section with `pip` commands).
- Install other requirements: `pip install -r requirements.txt`

### Configuring Capreolus
 Capreolus uses environment variables to indicate where outputs should be stored and where document inputs can be found. Consult the list below to determine which variables should be set. [Set these environment variables](https://opensource.com/article/19/8/what-are-environment-variables) either on the fly (`export CAPREOLUS_RESULTS=...`) before running Capreolus or by editing your shell's initialization files (e.g., `~/.bashrc` or `~/.zshrc`).
- `CAPREOLUS_RESULTS`: directory where results are stored (default: `~/.capreolus/results/`)
- `CAPREOLUS_CACHE`: directory where cache files are stored (default: `~/.capreolus/cache/`)
- `CAPREOLUS_LOGGING`: Indicates the logging level: `DEBUG`, `INFO` (default), `WARN` or `ERROR`
- `CUDA_VISIBLE_DEVICES`: Indicates GPUs available to PyTorch, starting from 0. For example, setting to '1' will use the system's 2nd GPU (as numbered by `nvidia-smi`). Set to "" (an empty string) to force CPU.

To avoid confusion and failed experiments due to limited disk space, we recommend always setting `CAPREOLUS_RESULTS` and `CAPREOLUS_CACHE` rather than relying on the default behavior. Typically, `CUDA_VISIBLE_DEVICES` is set immediately before running an experiment (e.g., to run several separate experiments on different GPUs in parallel).

You're now ready to run `capreolus`.
