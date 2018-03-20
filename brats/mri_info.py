import glob
import os
import nibabel as nib

def get_mri_info(img_folder):
    for img in glob.glob(img_folder):
        print(img)


if __name__=='__main__':
    get_mri_info('/mnt/960EVO/workspace/3DUnetCNN-fork/brats/data/preprocessed_test/test/*/*')