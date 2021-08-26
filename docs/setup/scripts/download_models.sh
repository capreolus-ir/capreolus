model=$1

if [ -z "$model" ]; then
        model="bert-base-uncased"
        echo "Warning: Didn't find input model, download $model by default"
fi

files=("config.json" "pytorch_model.bin" "tf_model.h5" "tokenizer.json" "tokenizer_config.json" "vocab.json" "vocab.txt" "merges.txt")

echo "Downloading $model..."
mkdir -p $model
rm -f $model/*

for file in "${files[@]}"
do
        url="https://huggingface.co/$model/resolve/main/$file"
        wget $url -P $model
        echo $url
done