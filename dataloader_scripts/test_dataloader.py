import os
import glob
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import random
import tqdm


def vol_2_5D_extractor_low(batch):
    patch_list = []
    d, w, h = batch.shape
    for xd in np.arange(0, d - patch_depth + 1, 1):
        patch_list.append(batch[xd:xd + patch_depth, :, :])

    return np.asarray(patch_list)



def normalize_train(batch):
    return np.squeeze(batch)


class LowDosePETData_test(Dataset):
    """
    This a basic class to load pre-generated PET and MR images slice by slice.

    idx_lists: a list containing subject indexes corresponding to INDEX2DATA_MAPPING_2D
    """

    def __init__(self, num_slices=9, next_patient_idx=0, mode='tra') -> None:
        super().__init__()
        global ROOT_PATH
        global ROOT_PATH_unet_guided
        global noise_level

        self.mode = mode
        self.next_patient_idx = next_patient_idx

        self.images = {}
        self.indexes = []
        subject_idx = 0
        global patch_depth
        patch_depth = num_slices
        print("getting ", patch_depth, " slices")

        training_data_list = "./test.txt"
        Train_Patient_List = []
        with open(training_data_list) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                Train_Patient_List.append(line.split(" ")[0])

        ROOT_PATH = './data/'
        ROOT_PATH_unet_guided = './data/'
        noise_level = ['1', '2', '5', '10', '25', '50']
        dose_scaling_factor = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5]

        Train_Patient_List = Train_Patient_List[self.next_patient_idx:self.next_patient_idx+1]

        for ID in tqdm.tqdm(Train_Patient_List):
            pet_images_hd_true = None
            for n_l in range(len(dose_scaling_factor)):
                path = ROOT_PATH + ID + "_full_dose.nii.gz"
                print(subject_idx, "Loading PET images ", path)
                pet_images_ld, pet_images_unet, pet_images_hd = self.load_images_from_idx(ID, n_l, is_load_hd=(pet_images_hd_true is None))

                if pet_images_hd_true is None:
                    pet_images_hd_true = np.copy(pet_images_hd)

                self.images[subject_idx] = (pet_images_ld,
                                            pet_images_unet,
                                            pet_images_hd_true,
                                            pet_images_hd_true.shape[0])
                for slice_idx in range(pet_images_hd_true.shape[0]):
                    self.indexes.append([subject_idx, slice_idx])
                subject_idx = subject_idx + 1

                del pet_images_hd

    @staticmethod
    def load_images_from_idx(ID, n_l, is_load_hd=True):
        """
        This functon returns PET images given the subject idx.

        idx: subject idx.
        """

        path_low_dose_unet = ROOT_PATH_unet_guided + '/' + ID + '_unet_' + noise_level[n_l] + ".nii.gz"
        pet_images_low_dose_unet = np.asarray(nib.load(path_low_dose_unet).dataobj)

        pet_images_low_dose_unet = np.expand_dims(pet_images_low_dose_unet, 1)

        path_low_dose = ROOT_PATH + '/' + ID + "_low_dose_" + noise_level[n_l] + ".nii.gz"
        pet_images_low_dose = np.asarray(nib.load(path_low_dose).dataobj)

        print("low: ", pet_images_low_dose.shape)
        print("unet: ", pet_images_low_dose_unet.shape)

        pet_images_low_dose = vol_2_5D_extractor_low(pet_images_low_dose)

        slice_diff = pet_images_low_dose_unet.shape[0] - pet_images_low_dose.shape[0]
        if slice_diff!=0 and slice_diff>0:
            pet_images_low_dose = np.append(pet_images_low_dose, np.zeros([slice_diff, patch_depth,
                                                                           pet_images_low_dose.shape[2],
                                                                           pet_images_low_dose.shape[3]],
                                                                          dtype=np.float32), axis=0)

        print("low: ", pet_images_low_dose.shape)


        if is_load_hd:
            path_high_dose = ROOT_PATH + ID + "_full_dose.nii.gz"
            pet_images_high_dose = np.asarray(nib.load(path_high_dose).dataobj)
            pet_images_high_dose = np.expand_dims(pet_images_high_dose, 1)


        else:
            pet_images_high_dose = None

        return pet_images_low_dose, pet_images_low_dose_unet, pet_images_high_dose

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        subject_idx, slice_idx = self.indexes[item]

        pet_images_ld = self.images[subject_idx][0][slice_idx]
        pet_images_unet = self.images[subject_idx][1][slice_idx]
        pet_images_hd = self.images[subject_idx][2][slice_idx]
        num_slices = self.images[subject_idx][3]

        if self.mode == 'tra':
            return pet_images_hd, {'low_res': pet_images_ld, 'unet': pet_images_unet, "num_slices": num_slices}

