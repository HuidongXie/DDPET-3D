import os
import glob
import numpy as np
#from utility.utility import minmax_normalization
from torch.utils.data import Dataset
import nibabel as nib
import random
import tqdm

ROOT_PATH = 'YOUR_PATH'
noise_level = ['1','2','5','10','25','50']


def vol_2_5D_extractor_low(batch):
    patch_list = []
    d, w, h = batch.shape
    for xd in np.arange(0, d - patch_depth + 1, 1):
        patch_list.append(batch[xd:xd + patch_depth, :, :])

    return np.asarray(patch_list)



def normalize_train(batch):
    return np.squeeze(batch)


class LowDosePETData(Dataset):
    """
    This a basic class to load pre-generated PET and MR images slice by slice.

    idx_lists: a list containing subject indexes corresponding to INDEX2DATA_MAPPING_2D
    """

    def __init__(self, num_slices=9, mode='tra') -> None:
        super().__init__()

        self.mode = mode

        self.images = {}
        self.indexes = []
        subject_idx = 0

        global patch_depth
        patch_depth = num_slices

        training_data_list = "./train_patient_list.txt"
        Train_Patient_List = []
        with open(training_data_list) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                Train_Patient_List.append(line.split(" ")[0])

        Train_Patient_List = Train_Patient_List
        random.shuffle(Train_Patient_List)
        Train_Patient_List = Train_Patient_List[:1]

        for ID in tqdm.tqdm(Train_Patient_List):
            path = ROOT_PATH + ID + "_countlevel_" + noise_level[0] + ".nii.gz"
            print(subject_idx, "Loading PET images ", path)

            pet_images_hd_true = None
            for n_l in range(len(noise_level)):
                pet_images_ld, pet_images_hd = self.load_images_from_idx(ID, n_l,
                                                                         is_load_hd=(pet_images_hd_true is None))

                if pet_images_hd_true is None:
                    pet_images_hd_true = np.copy(pet_images_hd)

                self.images[subject_idx] = (pet_images_ld,
                                            pet_images_hd_true)
                print(pet_images_hd_true.shape[0])
                for slice_idx in range(pet_images_hd_true.shape[0]):
                    self.indexes.append([subject_idx, slice_idx, slice_idx/pet_images_hd_true.shape[0]])
                subject_idx = subject_idx + 1

                del pet_images_hd

    @staticmethod
    def load_images_from_idx(ID, n_l, is_load_hd=True):
        """
        This functon returns PET images given the subject idx.

        idx: subject idx.
        """
        path_low_dose = ROOT_PATH + ID + "_countlevel_" + noise_level[n_l] + ".nii.gz"
        pet_images_low_dose = np.transpose(np.asarray(nib.load(path_low_dose).dataobj), [2, 0, 1])
        pet_images_low_dose = vol_2_5D_extractor_low(pet_images_low_dose)


        if is_load_hd:
            path_high_dose = ROOT_PATH + ID + "_countlevel_" + '100' + ".nii.gz"
            pet_images_high_dose = np.transpose(np.asarray(nib.load(path_high_dose).dataobj), [2, 0, 1])

            pet_images_high_dose = pet_images_high_dose[1+(patch_depth-3)//2:-(1+(patch_depth-3)//2)]
            pet_images_high_dose = np.expand_dims(pet_images_high_dose, 1)
            # print(pet_images_high_dose.shape)

        else:
            pet_images_high_dose = None

        return pet_images_low_dose, pet_images_high_dose

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        subject_idx, slice_idx, slice_idx_norm = self.indexes[item]

        pet_images_ld = self.images[subject_idx][0][slice_idx]
        pet_images_hd = self.images[subject_idx][1][slice_idx]

        if self.mode == 'tra':
            return pet_images_hd, {'low_res': pet_images_ld}

