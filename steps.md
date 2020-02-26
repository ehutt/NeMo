# Setup 

## Setting up the EC2 Environment 
```
conda create -n nvidia-env python=3.6 
conda activate nvidia-env
# to support adding conda env to jupyter notebooks:
conda install -c anaconda ipykernel
```
Check that CUDA version >= 10.0 
```
nvcc --version
```
Install Pytorch version 1.?
```
conda install -c anaconda pytorch-gpu
```
Install Apex 
```
git clone https://github.com/NVIDIA/apex
cd apex/
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
Clone and initialize NeMo repo forked from NVIDIA 
```
git clone https://github.com/ehutt/NeMo.git
cd NeMo/
pip install nemo_toolkit
# if ERROR: torchvision 0.5.0 has requirement torch==1.4.0, but you'll have torch 1.3.1 which is incompatible.
pip install torch==1.4.0
pip install nemo_nlp
# test installation
./reinstall.sh
python -m unittest tests/*.py
# sphinx.errors.ThemeError: sphinx_rtd_theme is no longer a hard dependency since version 1.4.0. Please install it manually.(pip install sphinx_rtd_theme)
pip install sphinx_rtd_theme
```

Get SQuad Datasets 
```
python NeMo/examples/nlp/question_answering/get_squad.py --destDir data
```

Copy model checkpoints 
``` 
(local)$ scp path/to/checkpt/zip ubuntu@ec2-ip:path/to/models
```
To run only evaluation on pretrained question answering checkpoints on 1 GPU with ground-truth data:
```
cd NeMo/examples/nlp/question_answering
eval_file=/home/ubuntu/data/squad/v2.0/dev-v2.0.json
model_dir=/home/ubuntu/models/bert-large-uncased_ckpt

python question_answering_squad.py \
--eval_file $eval_file \
--pretrained_model_name bert-large-uncased \
--checkpoint_dir $model_dir \
--do_lower_case \
--mode eval
```

Pretrained Model Name Options: 
```
--pretrained_model_name {albert-base-v1,albert-large-v1,albert-xlarge-v1,albert-xxlarge-v1,albert-base-v2,albert-large-v2,albert-xlarge-v2,albert-xxlarge-v2,roberta-base,roberta-large,roberta-large-mnli,distilroberta-base,roberta-base-openai-detector,roberta-large-openai-detector,bert-base-uncased,bert-large-uncased,bert-base-cased,bert-large-cased,bert-base-multilingual-uncased,bert-base-multilingual-cased,bert-base-chinese,bert-base-german-cased,bert-large-uncased-whole-word-masking,bert-large-cased-whole-word-masking,bert-large-uncased-whole-word-masking-finetuned-squad,bert-large-cased-whole-word-masking-finetuned-squad,bert-base-cased-finetuned-mrpc,bert-base-german-dbmdz-cased,bert-base-german-dbmdz-uncased,bert-base-japanese,bert-base-japanese-whole-word-masking,bert-base-japanese-char,bert-base-japanese-char-whole-word-masking,bert-base-finnish-cased-v1,bert-base-finnish-uncased-v1,bert-base-dutch-cased}]
```
