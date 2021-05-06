################################
# Environment
################################
setup_dir=$1
if [ -z $setup_dir ]; then
	echo "Error: Setup directory required"
	exit
fi

env_name=capreolus
config_file="$setup_dir/setup_capreolus_on_cc.bash"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda create --name $env_name python=3.7.9 -y
conda activate $env_name

# Java
setup_path="$setup_dir/setup"
download_path="$setup_path/download"

mkdir -p $setup_path
rm -rf $setup_path/*
mkdir -p $download_path

echo "----------"
conda env list
echo "----------"
pip install gdown
conda install gdown
gdown "https://drive.google.com/uc?id=1iUhEnCwO9SDCCbNrA9dc6fe5phtcfLU9" -O "$download_path/jdk-11.0.9_linux-x64_bin.tar.gz"
gdown "https://drive.google.com/uc?id=1gLiwlxq8YhJeCth-y6O_bAT49ATLRw-a" -O "$download_path/libs.tar.gz"
tar -xzf "$download_path/jdk-11.0.9_linux-x64_bin.tar.gz" -C "$setup_path"
tar -xzf "$download_path/libs.tar.gz" -C "$setup_path"

server=$(uname -n)
lib_home=$setup_path/libs
java_home=$setup_path/jdk-11.0.9
echo "export JAVA_HOME=$java_home" > $config_file
echo "export PATH=$java_home/bin/:\$PATH" >> $config_file

if [[ "$server" == "gra-login"* ]]; then
        echo "export LD_LIBRARY_PATH=$lib_home/lib:\$LD_LIBRARY_PATH" >> $config_file
elif [[ "$server" == "cedar"* ]]; then
        echo "export LD_LIBRARY_PATH=$lib_home/libjpeg/lib:$lib_home/lib:$lib_home/openjpeg2-2.3.1/usr/lib64:$lib_home/libtiff-4.0.3/usr/lib64:$lib_home/jbigkit-libs-2.0/usr/lib64:\$LD_LIBRARY_PATH" >> $config_file
elif [[ "$server" == "beluga"* ]]; then
        echo "beluga"
        echo "export LD_LIBRARY_PATH=$lib_home/lib:$lib_home/openjpeg2-2.3.1/usr/lib64:\$LD_LIBRARY_PATH"  >> $config_file
else
        echo "Error: Unexpected server"
        exit
fi


################################
# some 'pre-download' huggingface models
# since graham and beluga has no internet access
################################
model_dir=hugginface_models
mkdir -p $model_dir
rm -rf $model_dir/*

hugginface_models=("bert-base-uncased" "bert-large-uncased" "Capreolus/bert-base-msmarco")
for model in "${hugginface_models[@]}"
do
	sh ./scripts/download_models.sh $model
	mv $model $model_dir
done


################################
# Python package
################################
source $config_file

conda install -c conda-forge tensorflow=2.3.0 -y
conda install -c conda-forge --file ./scripts/cc-requirements.conda.txt -y
pip install -r ./scripts/cc-requirements.pip.txt 
pip install --no-deps -r ./scripts/cc-requirements-no-deps.txt

env_path=$(dirname $(dirname $(which conda)))
cp $lib_home/pytrec_eval_ext.cpython-37m-x86_64-linux-gnu.so $env_path/envs/$env_name/lib/python3.7/site-packages/pytrec_eval_ext.cpython-37m-x86_64-linux-gnu.so

