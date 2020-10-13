# Installation

This section covers installing Capreolus via its pip package or from source.


## Prerequisites
Capreolus requires both Python 3.7+ and Java 11.

The easiest way to install these dependencies is by using the *Conda package manager* as described in this [guide to installing Miniconda and Python 3](https://gist.github.com/andrewyates/970c570411c4a36785f6c0e9362eb1eb).
We recommend installing Capreolus into its own Conda environment using the provided `environment.yml` file:
1. `wget https://raw.githubusercontent.com/capreolus-ir/capreolus/master/environment.yml`
2. `conda env create --name MyCapreolus -f environment.yml`
3. `conda activate MyCapreolus`


## Installing Capreolus via pip
0. Activate the appropriate environment (if using conda): `conda activate MyCapreolus`
1. `pip install capreolus`
2. You can now use Capreolus on the command line via the `capreolus` command
 
## Configuring Capreolus
 Capreolus uses environment variables to indicate where outputs should be stored and where document inputs can be found. Consult the list below to determine which variables should be set. [Set these environment variables](https://opensource.com/article/19/8/what-are-environment-variables) either on the fly (`export CAPREOLUS_RESULTS=...`) before running Capreolus or by editing your shell's initialization files (e.g., `~/.bashrc` or `~/.zshrc`).
- `CAPREOLUS_RESULTS`: directory where results are stored (default: `~/.capreolus/results/`)
- `CAPREOLUS_CACHE`: directory where cache files are stored (default: `~/.capreolus/cache/`)
- `CAPREOLUS_LOGGING`: Indicates the logging level: `DEBUG`, `INFO` (default), `WARN` or `ERROR`
- `CUDA_VISIBLE_DEVICES`: Indicates GPUs available to PyTorch, starting from 0. For example, setting to '1' will use the system's 2nd GPU (as numbered by `nvidia-smi`). Set to "" (an empty string) to force CPU. 

To avoid confusion and failed experiments due to limited disk space, we recommend always setting `CAPREOLUS_RESULTS` and `CAPREOLUS_CACHE` rather than relying on the default behavior. Typically, `CUDA_VISIBLE_DEVICES` is set immediately before running an experiment (e.g., to run several separate experiments on different GPUs in parallel).

You're now ready to run `capreolus`.


<br/>
<hr/>
<br/>

## Alternate installation approaches
This section describes alternate ways to install Capreolus. We strongly recommend installing via pip when possible (as described above).

### Installing Capreolus from source
1. Clone the Capreolus repository: `git clone https://github.com/capreolus-ir/capreolus`
2. You should now have a `capreolus` folder that contains various files as well as another `capreolus` folder, which contains the actual capreolus Python package. This is a [common layout for Python packages](https://python-packaging.readthedocs.io/en/latest/minimal.html); the inside folder (i.e., `capreolus/capreolus`) corresponds to the Python package.
3. `cd capreolus`
4. [Install PyTorch](https://pytorch.org/get-started/locally/)
5. `pip install -r requirements.txt`
6. You can now use Capreolus on the command line via the `scripts/capreolus` command. Note that this only works from the outer `capreolus` directory; you will need to [adjust PYTHONPATH](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html).

