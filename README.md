# Blindly Assess Quality of In-the-Wild Videos via Quality-aware Pre-training and Motion Perception

## Description
Source code for the following paper:

- Bowen Li, Weixia Zhang, Meng Tian, Guangtao Zhai, and Xianpei Wang. [Blindly Assess Quality of In-the-Wild Videos via Quality-aware Pre-training and Motion Perception] [[arxiv version]](https://arxiv.org/abs/2108.08505)
![Framework](Overall_Framework.png)

## Usage
### Install Requirements
```bash
conda create -n reproducibleresearch pip python=3.6
source activate reproducibleresearch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# source deactive
```
Note: Make sure that the CUDA version is consistent. If you have any installation problems, please find the details of error information in `*.log` file.

## Download VQA datasets
Download the [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html), [CVD2014](https://www.mv.helsinki.fi/home/msjnuuti/CVD2014/), [LIVE-Qualcomm](http://live.ece.utexas.edu/research/incaptureDatabase/index.html), [LIVE-VQC](http://live.ece.utexas.edu/research/LIVEVQC/index.html), [YouTube-UGC](https://github.com/vztu/BVQA_Benchmark), and [LSVQ](https://github.com/baidut/PatchVQ) datasets. Then, run the following `ln` commands in the root of this project.

```bash
ln -s KoNViD-1k_path KoNViD-1k # KoNViD-1k_path is your path to the KoNViD-1k dataset
ln -s CVD2014_path CVD2014 # CVD2014_path is your path to the CVD2014 dataset
ln -s LIVE-Qualcomm_path LIVE-Qualcomm # LIVE-Qualcomm_path is your path to the LIVE-Qualcomm dataset
ln -s LIVE-VQC_path LIVE-VQC # LIVE-VQC_path is your path to the LIVE-VQC dataset
ln -s YouTube-UGC_path YouTube-UGC # YouTube-UGC_path is your path to the YouTube-UGC dataset
ln -s LSVQ_path LSVQ # LSVQ_path is your path to the LSVQ dataset
``` 

## Spatial Fearure: Transfer knowledge from quality-aware pre-training
#### Sampling image pairs from multiple databases
data_all.m  
#### Combining the sampled pairs to form the training set
combine_train.m  
#### Training on multiple databases for 10 sessions
python Main.py --train True --network basecnn --representation BCNN --ranking True --fidelity True --std_modeling True --std_loss True --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --max_epochs2 12 


## Motion Fearure: Transfer knowledge from motion perception
python Main.py --train True --network basecnn --representation BCNN --ranking True --fidelity True --std_modeling True --std_loss True --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --max_epochs2 12

### Training and Evaluating on Multiple Datasets

```bash
# Feature extraction
CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8
CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=LIVE-VQC --frame_batch_size=8
# Training, intra-dataset evaluation, for example 
chmod 777 job.sh
./job.sh -g 0 -d K -d C -d L > KCL-mixed-exp-0-10-1e-4-32-40.log 2>&1 &
# Cross-dataset evaluation (after training), for example
chmod 777 cross_job.sh
./cross_job.sh -g 1 -d K -d C -d L -c N -l mixed > KCLtrained-crossN-mixed-exp-0-10.log 2>&1 &
```

### Test Demo

The model weights provided in `models/MDTVSFA.pt` are the saved weights when running the 9-th split of KoNViD-1k, CVD2014, and LIVE-Qualcomm.
```bash
python test_demo.py --model_path=models/MDTVSFA.pt --video_path=data/test.mp4
```

### Contact
Dingquan Li, dingquanli AT pku DOT edu DOT cn.

