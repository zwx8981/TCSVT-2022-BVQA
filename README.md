# Blindly Assess Quality of In-the-Wild Videos via Quality-aware Pre-training and Motion Perception

## Description
Source code for the following paper:

- Bowen Li, Weixia Zhang, Meng Tian, Guangtao Zhai, and Xianpei Wang. [Blindly Assess Quality of In-the-Wild Videos via Quality-aware Pre-training and Motion Perception] [[arxiv version]](https://arxiv.org/abs/2108.08505)
![Framework](Overall_Framework.png)

# Usage
## Install Requirements
```bash
conda create -n reproducibleresearch pip python=3.6
source activate reproducibleresearch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Download VQA Databases
Download the [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html), [CVD2014](https://www.mv.helsinki.fi/home/msjnuuti/CVD2014/), [LIVE-Qualcomm](http://live.ece.utexas.edu/research/incaptureDatabase/index.html), [LIVE-VQC](http://live.ece.utexas.edu/research/LIVEVQC/index.html), [YouTube-UGC](https://github.com/vztu/BVQA_Benchmark), and [LSVQ](https://github.com/baidut/PatchVQ) datasets. Then, run the following `ln` commands in the root of this project.

```bash
ln -s KoNViD-1k_path KoNViD-1k # KoNViD-1k_path is your path to the KoNViD-1k dataset
ln -s CVD2014_path CVD2014 # CVD2014_path is your path to the CVD2014 dataset
ln -s LIVE-Qualcomm_path LIVE-Qualcomm # LIVE-Qualcomm_path is your path to the LIVE-Qualcomm dataset
ln -s LIVE-VQC_path LIVE-VQC # LIVE-VQC_path is your path to the LIVE-VQC dataset
ln -s YouTube-UGC_path YouTube-UGC # YouTube-UGC_path is your path to the YouTube-UGC dataset
ln -s LSVQ_path LSVQ # LSVQ_path is your path to the LSVQ dataset
``` 

## Spatial Fearure: Transfer Knowledge from Quality-aware Pre-training
#### Sampling image pairs from multiple databases
data_all.m  
#### Combining the sampled pairs to form the training set
combine_train.m  
#### Training on multiple databases for 10 sessions
python Main.py --train True --network basecnn --representation NOTBCNN --ranking True --fidelity True --std_modeling True --std_loss True --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --max_epochs2 12 
#### Feature extraction
CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=64
CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8
CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=LIVE-VQC --frame_batch_size=8
CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=YouTube-UGC --frame_batch_size=8
CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=LSVQ --frame_batch_size=8

## Motion Fearure: Transfer Knowledge from Motion Perception

#### Prepare the pre-trained SlowFast model file in the directory "./MotionExtractor/checkpoints/Kinetics/"
[SlowFast_Model](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl)
#### Feature extraction
CUDA_VISIBLE_DEVICES=&gpu_id python CNNfeatures.py --database=&database --frame_batch_size=64


## Training and Evaluating on VQA Databases

```bash
# Training, under individual-dataset setting, for example 
python main.py --trained_datasets ['C'] --tested_datasets ['C']
# Training, under mixed-database setting, for example
python main.py --trained_datasets ['K', 'C', 'L', 'N'] --tested_datasets ['K', 'C', 'L', 'N']
```

## Test Demo

The model weights provided in `models/model` are the saved weights when best performing on CVD2014.
```bash
python test_demo.py --model_path=models/model --video_path=data/test.mp4
```

## Acknowledge

This work is heavily inspired by [MDTVSFA](https://github.com/lidq92/MDTVSFA) (Li—IJCV2021).
The model-based transfer learning for video features extraction using mainly follows the implementations of [UNIQUE - IQA domain](https://github.com/zwx8981/UNIQUE) (Zhang—TIP2021) and [SlowFast - Action Recognition domain](https://github.com/facebookresearch/SlowFast) (Feichtenhofer—ICCV2019).
Great appreciation for theit excellent works.

## Citation

Should you find this repo useful to your research, we sincerely appreciate it if you cite our paper :blush: :
```bash
@article{li2021blindly,
  title={Blindly Assess Quality of In-the-Wild Videos via Quality-aware Pre-training and Motion Perception},
  author={Li, Bowen and Zhang, Weixia and Tian, Meng and Zhai, Guangtao and Wang, Xianpei},
  journal={arXiv preprint arXiv:2108.08505},
  year={2021}
}
```
