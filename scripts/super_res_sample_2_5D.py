"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import nibabel as nib
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from torch.utils.data import DataLoader
from dataloader_scripts.dataset_test_2_5D import LowDosePETData_test
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_slices = 9
th.manual_seed(66)

def testing_dataloader_wrapper(dataset, args):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)
    while True:
        yield from data_loader


def main():
    args = create_argparser().parse_args()
    args.large_size = 320
    args.in_channels = num_slices

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    # data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)
    training_data = LowDosePETData_test(num_slices=args.in_channels)
    data = testing_dataloader_wrapper(training_data, args)

    logger.log("creating samples... number_samples", args.num_samples)
    #exit(0)
    all_images = []

    test_sample_idx = 0

    for num_patient in range(6):
        estimated_sample = np.zeros(shape=[672, args.large_size, args.large_size])
        estimated_low_res = np.zeros(shape=[672, args.large_size, args.large_size])
        estimated_high_res = np.zeros(shape=[672, args.large_size, args.large_size])
        for b in range(672):
            high_res, model_kwargs = next(data)
            # if b<350:
            #     continue
            print("testing: ", b, model_kwargs['low_res'].shape)

            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

            sample = diffusion.ddim_sample_loop(
                model,
                (args.batch_size, 1, args.large_size, args.large_size),
                noise=None,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            sample = np.squeeze(sample.detach().cpu().numpy())
            high_res = np.squeeze(high_res.detach().cpu().numpy())
            los_res = np.squeeze(model_kwargs['low_res'].detach().cpu().numpy())[args.in_channels//2]

            estimated_sample[b] = sample
            estimated_low_res[b] = los_res
            estimated_high_res[b] = high_res

            test_sample_idx = test_sample_idx + 1


        estimated_high_res[estimated_high_res < 0] = 0.0
        estimated_low_res[estimated_low_res < 0] = 0.0
        estimated_sample[estimated_sample < 0] = 0.0
        estimated_sample = estimated_sample * (np.mean(estimated_low_res)/np.mean(estimated_sample))
        estimated_high_res = estimated_high_res * (np.mean(estimated_low_res)/np.mean(estimated_high_res))
        estimated_low_res = estimated_low_res * (np.mean(estimated_low_res)/np.mean(estimated_low_res))

        print("MAE In: ", np.mean(np.abs(estimated_low_res - estimated_high_res)))
        print("MAE diffusion: ", np.mean(np.abs(estimated_sample - estimated_high_res)))

        estimated = np.squeeze(estimated_sample)
        new_image = nib.Nifti1Image(estimated, affine=np.eye(4))
        f_n = "./test_2_5D_all_ddim_eta03/full_res_" + str(num_patient) + ".nii"
        nib.save(new_image, f_n)

        estimated = np.squeeze(estimated_high_res)
        new_image = nib.Nifti1Image(estimated, affine=np.eye(4))
        f_n = "./test_2_5D_all_ddim_eta03/full_real_" + str(num_patient) + ".nii"
        nib.save(new_image, f_n)

        estimated = np.squeeze(estimated_low_res)
        new_image = nib.Nifti1Image(estimated, affine=np.eye(4))
        f_n = "./test_2_5D_all_ddim_eta03/full_in_" + str(num_patient) + ".nii"
        nib.save(new_image, f_n)


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1000,
        batch_size=1,
        use_ddim=True,
        base_samples="",
        model_path="./models/model_2_5D_combine/ema_0.9999_105000.pt",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
