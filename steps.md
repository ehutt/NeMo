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
model_dir=/home/ubuntu/models/squadv2.0_roberta-large_from_huggingface_ckpt
model_name=roberta-large #see options below 

python question_answering_squad.py \
--eval_file $eval_file \
--pretrained_model_name $model_name \
--checkpoint_dir $model_dir \
--do_lower_case \
--version_2_with_negative \
--mode eval \
--no_data_cache
```

To finetune Squad v2.0 on pretrained BERT large uncased on 1 GPU:
```
mkdir tune_output

tune_out_dir=tune_output
train_file=/home/ubuntu/data/squad/v2.0/train-v2.0.json
eval_file=/home/ubuntu/data/squad/v2.0/dev-v2.0.json
checkpoint=/home/ubuntu/models/bert-large-uncased_ckpt/bert_large_uncased.pt
model_name=bert-large-uncased

python question_answering_squad.py \
--train_file $train_file \
--eval_file $eval_file \
--work_dir $tune_out_dir \
--pretrained_model_name $model_name \
--bert_checkpoint $checkpoint \ 
--amp_opt_level "O1" \
--batch_size 24 \
--num_epochs 2 \
--lr_policy WarmupAnnealing \
--lr_warmup_proportion 0.0 \
--optimizer adam_w \
--weight_decay 0.0 \
--lr 3e-5 \
--do_lower_case \
--mode train_eval \
--version_2_with_negative 
```


Pretrained Model Name Options: 
```
--pretrained_model_name {albert-base-v1,albert-large-v1,albert-xlarge-v1,albert-xxlarge-v1,albert-base-v2,albert-large-v2,albert-xlarge-v2,albert-xxlarge-v2,roberta-base,roberta-large,roberta-large-mnli,distilroberta-base,roberta-base-openai-detector,roberta-large-openai-detector,bert-base-uncased,bert-large-uncased,bert-base-cased,bert-large-cased,bert-base-multilingual-uncased,bert-base-multilingual-cased,bert-base-chinese,bert-base-german-cased,bert-large-uncased-whole-word-masking,bert-large-cased-whole-word-masking,bert-large-uncased-whole-word-masking-finetuned-squad,bert-large-cased-whole-word-masking-finetuned-squad,bert-base-cased-finetuned-mrpc,bert-base-german-dbmdz-cased,bert-base-german-dbmdz-uncased,bert-base-japanese,bert-base-japanese-whole-word-masking,bert-base-japanese-char,bert-base-japanese-char-whole-word-masking,bert-base-finnish-cased-v1,bert-base-finnish-uncased-v1,bert-base-dutch-cased}]
```

## Model comparison 

| BERT Variation | Size | # Parameters | SQuad Version | Accuracy | F1 |
|----------------|-------|--------------|----------------|--------|---|
| Roberta | Large | 355M | SQuad 2.0 | 84.09 | 87.13 | 
| BERT | Large | 340M | SQuad 2.0 | 82.57 | 89.80 | 
| Albert v2 | Large | ? | SQuad 2.0 | 81.79 | 85.06 | 

### SQuad 1.1 Fine-Tuned Model Checkpoints 
* Albert 
  * XL v2 
  * Large v2 
  * Base v2 
* Roberta 
  * Large 
  * Base 
* BERT 
  * Large, uncased 
  * Base, uncased 
  * Based, cased 
