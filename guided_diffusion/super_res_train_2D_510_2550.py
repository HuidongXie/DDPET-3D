import os
import glob
import numpy as np
#from utility.utility import minmax_normalization
from torch.utils.data import Dataset
import nibabel as nib
import random
import tqdm

ROOT_PATH = '/data17/user/HUIDONG_data/fed_learn_data/shanghai_fullsize/'
noise_level_low = ['5','10']
noise_level_high = ['25','50']

patch_width = 64
patch_depth = 20

training_data_list = "./ShanghaiUExplorer_train.txt"
Train_Patient_List = []
with open(training_data_list) as f:
    while True:
        line = f.readline()
        if not line:
            break
        Train_Patient_List.append(line.split(" ")[0])

Train_Patient_List = Train_Patient_List
random.shuffle(Train_Patient_List)


def normalize_train(batch):
    batch = (batch - np.mean(batch)) / np.std(batch)
    batch[batch < 0.0] = 0.0
    return np.squeeze(batch)


class LowDosePETData(Dataset):
    """
    This a basic class to load pre-generated PET and MR images slice by slice.

    idx_lists: a list containing subject indexes corresponding to INDEX2DATA_MAPPING_2D
    """

    def __init__(self, mode='tra') -> None:
        super().__init__()

        self.mode = mode

        self.images = {}
        self.indexes = []
        subject_idx = 0
        for ID in tqdm.tqdm(Train_Patient_List):
            path = ROOT_PATH + ID + "_countlevel_" + noise_level_low[0] + ".nii"
            # path = np.asarray(nib.load("F:/Anonymous_ANO_20220219_1746459_165142_1_input.nii").dataobj)
            print(subject_idx, "Loading PET and MR images ", path)

            # self.images[subject_idx] = self.load_images_from_idx(subject_idx)
            for n_l in range(2):
                pet_images_ld, pet_images_hd = self.load_images_from_idx(ID, n_l)
                self.images[subject_idx] = (pet_images_ld, pet_images_hd)
                for slice_idx in range(pet_images_hd.shape[0]):
                    self.indexes.append([subject_idx, slice_idx])
                subject_idx = subject_idx + 1

    @staticmethod
    def load_images_from_idx(ID, n_l):
        """
        This functon returns PET and MR images given the subject idx.

        idx: subject idx.
        """
        path_low_dose = ROOT_PATH + ID + "_countlevel_" + noise_level_low[n_l] + ".nii"
        pet_images_low_dose = np.transpose(np.asarray(nib.load(path_low_dose).dataobj), [2, 0, 1])
        pet_images_low_dose = normalize_train(pet_images_low_dose)

        path_high_dose = ROOT_PATH + ID + "_countlevel_" + noise_level_low[n_l] + ".nii"
        pet_images_high_dose = np.transpose(np.asarray(nib.load(path_high_dose).dataobj), [2, 0, 1])
        pet_images_high_dose = normalize_train(pet_images_high_dose)

        pet_images_low_dose = pet_images_low_dose[:,20:-20,20:-20]
        pet_images_high_dose = pet_images_high_dose[:,20:-20,20:-20]

        # print(pet_images_low_dose.shape)
        # print(pet_images_high_dose.shape)
        # exit(0)

        return pet_images_low_dose, pet_images_high_dose

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        subject_idx, slice_idx = self.indexes[item]

        pet_images_ld = self.images[subject_idx][0][slice_idx]
        pet_images_hd = self.images[subject_idx][1][slice_idx]

        # print(pet_images_ld.shape)
        # print(pet_images_hd.shape)
        # exit(0)

        # if self.is_central_crop:
        #     pass
        #     # for zoom1:
        #     # pet_image = pet_image[98:-118, 108:-108]
        #     # mri_image = mri_image[98:-118, 108:-108]
        #
        #     # pet_image = pet_image[24: -64, 44: -44]
        #     # mri_image = mri_image[24: -64, 44: -44]

        pet_images_ld = np.expand_dims(pet_images_ld, 0)
        pet_images_hd = np.expand_dims(pet_images_hd, 0)

        if self.mode == 'tra':
            return pet_images_hd, {'low_res': pet_images_ld}

