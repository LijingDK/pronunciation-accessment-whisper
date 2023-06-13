# Enhancing Whisper Model for Pronunciation Assessment with Multi-Adapters

# Environment dependent
1.Kaldi (Data preparation related function script)[Github link](https://github.com/kaldi-asr/kaldi)  
2.Espnet-0.10.4  
3.Modify the installation address of espnet in the path.sh file  
## Installation  
### Set up kaldi environment  
git clone -b 5.4 https://github.com/kaldi-asr/kaldi.git kaldi  
cd kaldi/tools/; make; cd ../src; ./configure; make  
### Set up espnet environment
git clone -b v.0.10.4 https://github.com/espnet/espnet.git  
cd espnet/tools/        # change to tools folder  
ln -s {kaldi_root}      # Create link to Kaldi. e.g. ln -s home/theanhtran/kaldi/  
### Set up Conda environment
./setup_anaconda.sh anaconda espnet 3.7.9   # Create a anaconda environmetn - espnet with Python 3.7.9  
make TH_VERSION=1.8.0 CUDA_VERSION=10.2     # Install Pytorch and CUDA  
. ./activate_python.sh; python3 check_install.py  # Check the installation  
conda install torchvision==0.9.0 torchaudio==0.8.0 -c pytorch  
### Set your own execution environment
Open path.sh file, change your espnet directory  
e.g. MAIN_ROOT=MAIN_ROOT=/mnt/data/lj/oriange/espnet-master
