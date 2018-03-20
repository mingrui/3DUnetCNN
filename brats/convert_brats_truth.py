import glob
import os
import warnings
import shutil
import nibabel as nib

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

from brats.tiantan_preprocess import convert_image_format
import brats.predict
from brats.config import config_one

from keras_contrib.layers import Deconvolution3D

import nipype.interfaces.fsl as fsl

def normalize_data(img_data):
    mean = np.mean(img_data, axis=(0, 1, 2))
    std = np.std(img_data, axis=(0, 1, 2))

    img_data -= mean
    img_data /= std

    #print(np.mean(img_data, axis=(0, 1, 2)))
    #print(np.std(img_data, axis=(0, 1, 2)))

    return img_data



def test_convert_image_to_nii():
    t2_path = '/mnt/960EVO/workspace/3DUnetCNN-fork/brats/data/predict_test/IDH_158/t2.nii.gz'
    normalized_data = '/mnt/960EVO/workspace/3DUnetCNN-fork/brats/data/predict_test/IDH_158/t2_normalized.nii.gz'
    out_path = '/mnt/960EVO/workspace/3DUnetCNN-fork/brats/data/predict_test/IDH_158/stripped.nii.gz'
    data_path = '/mnt/960EVO/workspace/3DUnetCNN-fork/brats/data/predict_test/IDH_158/data_t2.nii.gz'
    img = nib.load(t2_path)
    img_data = img.get_data()
    print(img_data.shape)

    # normalize data
    img_data = normalize_data(img_data)
    new_img =  nib.Nifti1Image(img_data,affine=img.affine)
    nib.save(new_img, normalized_data)

    #img = nib.load(data_path)
    #img_data = img.get_data()
    #print(np.mean(img_data, axis=(0, 1, 2)))
    #print(np.std(img_data, axis=(0, 1, 2)))
    '''shape = img_data.shape
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            for z in range(0, shape[2]):
                if img_data[x, y, z] > 1:
                    print(img_data[x, y ,z])
    '''

    btr = fsl.BET()
    btr.inputs.in_file = normalized_data
    btr.inputs.frac = 0.4
    btr.inputs.out_file = out_path
    res = btr.run()



def convert_brats_truth(truth_path, out_path):
    img = nib.load(truth_path)
    img_data = img.get_data()

    img_data[img_data > 1] = 1

    new_img = nib.Nifti1Image(img_data, affine=np.eye(4))
    nib.save(new_img, out_path)

def batch_convert_brats_truth():
    parent = '/media/mingrui/960EVO/workspace/3DUnetCNN-fork/brats/data/brats_2017_preprocessed/*/*'
    for subject_dir in glob.glob(parent):
        print(subject_dir)
        truth_file = os.path.join(subject_dir, "truth.nii.gz")
        out_file = os.path.join(subject_dir, "truth_T2.nii.gz")
        convert_brats_truth(truth_file, out_file)

def rename_one():
    parent = '/media/mingrui/960EVO/workspace/3DUnetCNN-fork/brats/data/brats_2017_preprocessed/*/*'
    for subject_dir in glob.glob(parent):
        print(subject_dir)
        os.rename(os.path.join(subject_dir, "truth.nii.gz"),os.path.join("truth_original.nii.gz"))

def rename_two():
    parent = '/media/mingrui/960EVO/workspace/3DUnetCNN-fork/brats/data/brats_2017_preprocessed/*/*'
    for subject_dir in glob.glob(parent):
        print(subject_dir)
        os.rename(os.path.join(subject_dir, "truth_T2.nii.gz"), os.path.join(subject_dir ,"truth.nii.gz"))

if __name__ == "__main__":
    print("testing: ")
    test_convert_image_to_nii()
    #batch_convert_brats_truth()
    #rename_one()
    #rename_two()

