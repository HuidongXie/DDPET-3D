# DDPET-3D

This is the code database for DDPET-3D (Dose-aware Diffusion Model for 3D Low-dose PET: Multi-institutional Validation with Reader Study and Real Low-dose Data): https://arxiv.org/abs/2405.12996

Training code ./scripts/super_res_train_2_5D_combine.py

Testing code ./scripts/super_res_sample_2_5D.py

Deep learning models reported in this work were built using PyTorch library v1.7.1 (Meta AI). The custom codes were written in Python v3.6.9. Dependent packages include Numpy v1.16.4, Nibabel v3.2.1, mpi4py v3.0.3, and blobfile v0.11.0. Statistical testings were performed using MATLAB (R2023b). PET image reconstructions were performed using clinical software provided by Siemens Healthineers and United Imaging Healthcare.

If you found this code or our work useful, please cite us.
```
@misc{xie_dose-aware_2024,
	title = {Dose-aware {Diffusion} {Model} for {3D} {Low}-dose {PET}: {Multi}-institutional {Validation} with {Reader} {Study} and {Real} {Low}-dose {Data}},
	url = {http://arxiv.org/abs/2405.12996},
	doi = {10.48550/arXiv.2405.12996},
	urldate = {2024-06-03},
	publisher = {arXiv},
	author = {Xie, Huidong and Gan, Weijie and Zhou, Bo and Chen, Ming-Kai and Kulon, Michal and Boustani, Annemarie and Spencer, Benjamin A. and Bayerlein, Reimund and Chen, Xiongchao and Liu, Qiong and Guo, Xueqi and Xia, Menghua and Zhou, Yinchi and Liu, Hui and Guo, Liang and An, Hongyu and Kamilov, Ulugbek S. and Wang, Hanzhong and Li, Biao and Rominger, Axel and Shi, Kuangyu and Wang, Ge and Badawi, Ramsey D. and Liu, Chi},
	month = may,
	year = {2024},
}
```

Contact: Huidong.Xie@yale.edu
Code adapted from https://github.com/openai/guided-diffusion
