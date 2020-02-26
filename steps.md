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
