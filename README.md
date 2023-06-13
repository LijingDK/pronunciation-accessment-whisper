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
Clone or download this repository and set it as working directory，open path.sh file, change your espnet directory.  
```
e.g. MAIN_ROOT=MAIN_ROOT=/mnt/data/lj/oriange/espnet-master  
```
### Whisper
install Whisper
```
pip install git+https://github.com/openai/whisper.git 
```
Load the pretrained model
```
import whisper
whisper.load("base.en")
whisper.load("small.en")
whisper.load("medium.en")
whisper.load("large-v2")
```
# Instructions for use
## Data preparation  
1.If you are not interested in kaldi or you are not interested in the generation of alignment information, you can skip this step and proceed to the next step, we have provided Alignment information in the [dump](link: https://pan.baidu.com/s/1ZbTqaC5E8eOzDtEHQg8EKg extract code: 7777).
2.Downlod the speechocean76 dataset from [speechocean762](https://www.openslr.org/101). Use your own Kaldi ASR model or public Kaldi ASR model (e.g., the Librispeech ASR Chain Model we used) and run Kaldi GOP recipe following its instruction. After the run finishs, you should see the perfrmance of the baseline model with the ASR model you use.

Then, extract the GOP features from the intermediate files of the Kaldi GOP recipe run.

1.All the data used in the experiment are stored in the directory. 
2.In order to better reproduce my experimental results, you can download the data set first, and then put the decompressed file directly under the 'pronunciation_whisper-main' main directory. 
3.Other files can remain unchanged, you can use it directly.
## Pronunciation Evaluation System
1.Before running, you need to move the corresponding files of espnet2 to the directory corresponding to your 'espnet/espnet2' directory.
```
espnet_path=your_espnet_path
cd pronunciation_whisper-main/espnet2
cp -r espnet2/asr/encoder/whisper_encoder_gop.py ${espnet_path}/espnet2/asr/encoder
cp -r espnet2/asr/espnet_gop_multitask_model_whisper_adapter.py ${espnet_path}/espnet2/asr
cp -r espnet2/bin/gop_whisper_adapter.py ${espnet_path}/espnet2/bin
cp -r espnet2/tasks/gop_whisper_adapter.py ${espnet_path}/espnet2/tasks
cp -r espnet2/train/trainer_gop.py ${espnet_path}/espnet2/train
```
## Run Training and Evaluation
Run the following script.
```
bash run_lm_multi_whisper_three_adapter.sh
```
Results, best model, and log will be saved in the specified in .exp_whisper/
