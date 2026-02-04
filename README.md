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
@misc{xie2025doseawarediffusionmodel3d,
      title={Dose-aware Diffusion Model for 3D PET Image Denoising: Multi-institutional Validation with Reader Study and Real Low-dose Data}, 
      author={Huidong Xie and Weijie Gan and Reimund Bayerlein and Bo Zhou and Ming-Kai Chen and Michal Kulon and Annemarie Boustani and Kuan-Yin Ko and Der-Shiun Wang and Benjamin A. Spencer and Wei Ji and Xiongchao Chen and Qiong Liu and Xueqi Guo and Menghua Xia and Yinchi Zhou and Hui Liu and Liang Guo and Hongyu An and Ulugbek S. Kamilov and Hanzhong Wang and Biao Li and Axel Rominger and Kuangyu Shi and Ge Wang and Ramsey D. Badawi and Chi Liu},
      year={2025},
      eprint={2405.12996},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2405.12996}, 
}
```

## Question
Email me at: huidong.xie@aya.yale.edu
