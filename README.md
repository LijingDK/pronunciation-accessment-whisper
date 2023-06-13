# Enhancing Whisper Model for Pronunciation Assessment with Multi-Adapters

# Environment dependent
1.Kaldi (Data preparation related function script)[Github link](https://github.com/kaldi-asr/kaldi)  
2.Espnet-0.10.4  
3.Modify the installation address of espnet in the path.sh file  
## Installation  
### Set up kaldi environment  
```
git clone -b 5.4 https://github.com/kaldi-asr/kaldi.git kaldi  
cd kaldi/tools/; make; cd ../src; ./configure; make  
```
### Set up espnet environment
```
git clone -b v.0.10.4 https://github.com/espnet/espnet.git  
cd espnet/tools/        # change to tools folder  
ln -s {kaldi_root}      # Create link to Kaldi. e.g. ln -s home00/lijing/kaldi/  
```
### Set up Conda environment  
```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh  # install conda
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -y -n  lj  python=3.8.16    # create a environment
conda activate lj
```
### Set your own execution environment
```
Open path.sh file, change your espnet directory  
e.g. MAIN_ROOT=MAIN_ROOT=/mnt/data/lj/oriange/espnet-master  
```
# Instructions for use
## Data preparation  
1.All the data used in the experiment are stored in the directory, in which train is used for training, valid is the verification set and test are used for testing respectively.```data```  
2.In order to better reproduce my experimental results, you can download the data set first, and then directly change the path in different sets in directory. 
3.if you
