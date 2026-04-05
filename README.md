# DDPET-3D

DDPET-3D implementation for **3D low-dose PET denoising**.

---

## Trained Models (and Test Cases)
Download trained models at:  
https://drive.google.com/drive/folders/1gySHcYCAlD-bNDdcsmVtI7PTn6ciKYX3?usp=sharing

A few anonymized testing cases are also provided.

> After downloading, place the checkpoints and test cases in the expected paths used by `test.py`

---

## Installation

### Create the conda environment
```bash
conda env create -f environment.yml
conda activate DDPET_3D
```

## Run the sample test (inference)
Run the sample test
```bash
CUDA_VISIBLE_DEVICES=0 python3 test.py
```
Results will be saved at: ./results/

## Training
Update the dataloader in: ./dataloader_scripts: train_dataloader.py
Run training:
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py
```

## Citation
```
@article{XIE2026104039,
title = {Dose-aware diffusion model for 3D PET image denoising: Multi-institutional validation with reader study and real low-dose data},
journal = {Medical Image Analysis},
volume = {111},
pages = {104039},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104039},
url = {https://www.sciencedirect.com/science/article/pii/S1361841526001088},
author = {Huidong Xie and Weijie Gan and Reimund Bayerlein and Bo Zhou and Ming-Kai Chen and Michal Kulon and Annemarie Boustani and Kuan-Yin Ko and Der-Shiun Wang and Benjamin A. Spencer and Wei Ji and Xiongchao Chen and Qiong Liu and Xueqi Guo and Menghua Xia and Yinchi Zhou and Hui Liu and Liang Guo and Hongyu An and Ulugbek S. Kamilov and Hanzhong Wang and Biao Li and Axel Rominger and Kuangyu Shi and Ge Wang and Ramsey D. Badawi and Chi Liu}
}
```

## Question
Email me at: huidong.xie@aya.yale.edu
