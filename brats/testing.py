import glob
import os
import warnings
import shutil
import nibabel as nib

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

from tiantan_preprocess import convert_image_format
import predict
from train import config

def test_convert_image_to_nii():
    image_path = '/media/mingrui/960EVO/workspace/3DUnetCNN/brats/data/tiantan_preprocessed/train/IDH_1/truth.nii.gz'
    nii_path = '/media/mingrui/960EVO/workspace/3DUnetCNN/brats/data/tiantan_preprocessed/train/IDH_1/nii_truth.nii.gz'
    convert_image_format(image_path, nii_path)
    image = nib.load(os.path.abspath(nii_path))

if __name__ == "__main__":
    print("testing: ")
    #test_convert_image_to_nii()
    predict.run_validation_case(0, "./brats/testing", "./brats/3d_unet_model.h5", config["hdf5_file"], config["validation_file"], config["training_modalities"])

