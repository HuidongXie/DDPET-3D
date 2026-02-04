"""
test the model
"""

import argparse
import nibabel as nib
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import time
import os

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from torch.utils.data import DataLoader
from dataloader_scripts.dataset_test_2_5D_w_dose import LowDosePETData_test
import os

num_slices = 31
th.manual_seed(66)

test_noise_levels = ['1','2','5','10','25','50']

save_test_data_path = "./results/"
####### read testing patient list
testing_data_list = "./test.txt"  #testing list
Test_Patient_List = []
with open(testing_data_list) as f:
    while True:
        line = f.readline()
        if not line:
            break
        Test_Patient_List.append(line.split(" ")[0])
####### read testing patient list

def testing_dataloader_wrapper(dataset, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True)
    while True:
        yield from data_loader


def main():
    args = create_argparser().parse_args()
    args.large_size = 320
    args.in_channels = num_slices
    args.use_dose_embed = True
    args.timestep_respacing="ddim25"

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    print(args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    model = th.nn.DataParallel(model)

    logger.log("loading data...")

    logger.log("creating samples... number_samples", args.num_samples)
    all_images = []

    test_sample_idx = 0

    input_gaussian_noise_1 = th.randn(size=(1, 1, args.large_size, args.large_size), device='cuda:0')
    input_gaussian_noise_1 = input_gaussian_noise_1 * th.ones(size=(args.batch_size,1,1,1), device='cuda:0')
    input_gaussian_noise_2 = th.randn(size=(1, 1, args.large_size, args.large_size), device='cuda:0')
    input_gaussian_noise_2 = input_gaussian_noise_2 * th.ones(size=(args.batch_size,1,1,1), device='cuda:0')

    for num_test_patient in range(674):
        file_not_exist = 0
        print(Test_Patient_List[num_test_patient])
        for nl in range(len(test_noise_levels)):
            f_n = save_test_data_path + "test_" + test_noise_levels[nl] + "/" + \
                  Test_Patient_List[num_test_patient] + "_" + test_noise_levels[nl] + "_results.nii.gz"
            print(f_n)
            if not os.path.isfile(f_n):
                file_not_exist = 1

        if file_not_exist == 0:
            print(Test_Patient_List[num_test_patient], " patient finished")
            continue

        training_data = LowDosePETData_test(num_slices=args.in_channels, next_patient_idx=num_test_patient)
        data = testing_dataloader_wrapper(training_data, args)
        for nl in range(len(test_noise_levels)):
            high_res, model_kwargs = next(data)
            print("testing: ", 0, model_kwargs['low_res'].shape)
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            unet_res = model_kwargs['unet']
            vol_slices = model_kwargs['num_slices'].detach().cpu().numpy()[0]
            print(vol_slices)
            model_kwargs.pop('unet')
            estimated_sample = np.zeros(shape=[vol_slices, args.large_size, args.large_size])
            estimated_low_res = np.zeros(shape=[vol_slices, args.large_size, args.large_size])
            estimated_high_res = np.zeros(shape=[vol_slices, args.large_size, args.large_size])
            estimated_unet = np.zeros(shape=[vol_slices, args.large_size, args.large_size])
            model_kwargs.pop('num_slices')

            start = time.time()
            for b in range(vol_slices//args.batch_size):
                if b != 0:
                    high_res, model_kwargs = next(data)
                    # if b<350:
                    #     continue
                    print("testing: ", num_test_patient, ' num ', b, " out of ", vol_slices//args.batch_size
                          , model_kwargs['low_res'].shape, " total slices ", vol_slices)

                    model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
                    unet_res = model_kwargs['unet']
                    model_kwargs.pop('unet')
                    model_kwargs.pop('num_slices')

                add_noise_step = 24

                test_img_1 = diffusion.q_sample(unet_res,
                                   t=th.tensor(add_noise_step, dtype=th.int32, device='cuda:0'), noise=input_gaussian_noise_1)

                test_img_2 = diffusion.q_sample(unet_res,
                                   t=th.tensor(add_noise_step, dtype=th.int32, device='cuda:0'), noise=input_gaussian_noise_2)


                device = next(model.parameters()).device
                img_1 = test_img_1
                img_2 = test_img_2
                indices = list(range(diffusion.num_timesteps))[::-1]
                print(indices)
                for i in indices:

                    t = th.tensor([i] * args.batch_size, device=device)
                    with th.no_grad():
                        if i % 5 == 0:
                            out_1 = diffusion.p_sample(
                                model,
                                img_1,
                                t,
                                clip_denoised=args.clip_denoised,
                                denoised_fn=None,
                                cond_fn=None,
                                model_kwargs=model_kwargs
                            )
                            if i == indices[0]:
                                out_2 = diffusion.p_sample(
                                    model,
                                    img_2,
                                    t,
                                    clip_denoised=args.clip_denoised,
                                    denoised_fn=None,
                                    cond_fn=None,
                                    model_kwargs=model_kwargs
                                )
                        else:
                            out_1 = diffusion.ddim_sample(
                                model,
                                img_1,
                                t,
                                clip_denoised=args.clip_denoised,
                                denoised_fn=None,
                                cond_fn=None,
                                model_kwargs=model_kwargs
                            )
                            if i == indices[0]:
                                out_2 = diffusion.ddim_sample(
                                    model,
                                    img_2,
                                    t,
                                    clip_denoised=args.clip_denoised,
                                    denoised_fn=None,
                                    cond_fn=None,
                                    model_kwargs=model_kwargs
                                )
                        # img_1 = out_1["sample"]
                        # img_2 = out_2["sample"]
                        if i == indices[0]:
                            img_1 = (out_1["sample"] + out_2["sample"]) / 2
                        else:
                            img_1 = out_1["sample"]
                sample = img_1

                sample = np.squeeze(sample.detach().cpu().numpy())
                high_res = np.squeeze(high_res.detach().cpu().numpy())
                los_res = np.squeeze(model_kwargs['low_res'].detach().cpu().numpy())[:,args.in_channels//2,:,:]

                print(sample.shape,high_res.shape, los_res.shape, np.squeeze(unet_res.detach().cpu().numpy()).shape)

                estimated_sample[b*args.batch_size:(b+1)*args.batch_size] = sample
                estimated_low_res[b*args.batch_size:(b+1)*args.batch_size] = los_res
                estimated_high_res[b*args.batch_size:(b+1)*args.batch_size] = high_res
                estimated_unet[b*args.batch_size:(b+1)*args.batch_size] = np.squeeze(unet_res.detach().cpu().numpy())

                test_sample_idx = test_sample_idx + 1

            end = time.time()
            print(end - start)


            estimated_high_res[estimated_high_res < 0] = 0.0
            estimated_low_res[estimated_low_res < 0] = 0.0
            estimated_sample[estimated_sample < 0] = 0.0
            estimated_unet[estimated_unet < 0] = 0.0
            estimated_sample[estimated_high_res == 0] = 0.0


            estimated = np.squeeze(estimated_sample)
            new_image = nib.Nifti1Image(estimated, affine=np.eye(4))
            f_n = save_test_data_path + "test_" + test_noise_levels[nl] + "/" + \
                  Test_Patient_List[num_test_patient] + "_" + test_noise_levels[nl] + "_results.nii.gz"
            print("write", f_n)
            nib.save(new_image, f_n)

            estimated = np.squeeze(estimated_high_res)
            new_image = nib.Nifti1Image(estimated, affine=np.eye(4))
            f_n = save_test_data_path + "test_" + test_noise_levels[nl] + "/" + \
                  Test_Patient_List[num_test_patient] + "_" + test_noise_levels[nl] + "_real.nii.gz"
            print("write", f_n)
            nib.save(new_image, f_n)

            estimated = np.squeeze(estimated_low_res)
            new_image = nib.Nifti1Image(estimated, affine=np.eye(4))
            f_n = save_test_data_path + "test_" + test_noise_levels[nl] + "/" + \
                  Test_Patient_List[num_test_patient] + "_" + test_noise_levels[nl] + "_input.nii.gz"
            print("write", f_n)
            nib.save(new_image, f_n)

            estimated = np.squeeze(estimated_unet)
            new_image = nib.Nifti1Image(estimated, affine=np.eye(4))
            f_n = save_test_data_path + "test_" + test_noise_levels[nl] + "/" + \
                  Test_Patient_List[num_test_patient] + "_" + test_noise_levels[nl] + "_unn.nii.gz"
            print("write", f_n)
            nib.save(new_image, f_n)



def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1000,
        batch_size=10,
        use_ddim=True,
        base_samples="",
        model_path="./models/pretrained_models/ema_0.9999_100000.pt" #UI data use this
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argpa