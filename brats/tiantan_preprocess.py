"""
Tools for converting, normalizing, and fixing the brats data.
"""

import glob
import os
import warnings
import shutil

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

import nibabel as nib

import re

from config import config_isensee
import argparse

config = config_isensee


def append_basename(in_file, append):
    dirname, basename = os.path.split(in_file)
    base, ext = basename.split(".", 1)
    return os.path.join(dirname, base + append + "." + ext)


def get_background_mask(in_folder, out_file, truth_name="tu"):
    """
    This function computes a common background mask for all of the data in a subject folder.
    :param in_folder: a subject folder from the BRATS dataset.
    :param out_file: an image containing a mask that is 1 where the image data for that subject contains the background.
    :param truth_name: how the truth file is labeled int he subject folder
    :return: the path to the out_file
    """
    background_image = None
    for name in ["t2"] + [truth_name]:
        image = sitk.ReadImage(get_image(in_folder, name))
        if background_image:
            print(0)
            if name == truth_name and not (image.GetOrigin() == background_image.GetOrigin()):
                image.SetOrigin(background_image.GetOrigin())
            background_image = sitk.And(image == 0, background_image)
        else:
            print(1)
            background_image = image == 0
    sitk.WriteImage(background_image, out_file)
    return os.path.abspath(out_file)


def convert_image_format(in_file, out_file):
    sitk.WriteImage(sitk.ReadImage(in_file), out_file)
    return out_file


def window_intensities(in_file, out_file, min_percent=1, max_percent=99):
    image = sitk.ReadImage(in_file)
    image_data = sitk.GetArrayFromImage(image)
    out_image = sitk.IntensityWindowing(image, np.percentile(image_data, min_percent), np.percentile(image_data,
                                                                                                     max_percent))
    sitk.WriteImage(out_image, out_file)
    return os.path.abspath(out_file)


def correct_bias(in_file, out_file):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT ${PATH}:/path/to/ants/bin)"))
        output_image = sitk.N4BiasFieldCorrection(sitk.ReadImage(in_file))
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)


def rescale(in_file, out_file, minimum=0, maximum=20000):
    image = sitk.ReadImage(in_file)
    sitk.WriteImage(sitk.RescaleIntensity(image, minimum, maximum), out_file)
    return os.path.abspath(out_file)


def get_image(subject_folder, name):
    file_card = os.path.join(subject_folder, "*" + name + "*.nii")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


def background_to_zero(in_file, background_file, out_file):
    sitk.WriteImage(sitk.Mask(sitk.ReadImage(in_file), sitk.ReadImage(background_file, sitk.sitkUInt8) == 0),
                    out_file)
    return out_file


def check_origin(in_file, in_file2):
    image = sitk.ReadImage(in_file)
    image2 = sitk.ReadImage(in_file2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_file)


def normalize_image(in_file, out_file, bias_correction=True):
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file


def convert_tiantan_folder(in_folder, out_folder, truth_name="tu",
                           no_bias_correction_modalities=None):
    for name in ["t2"]:
        image_file = get_image(in_folder, name)
        out_file = os.path.abspath(os.path.join(out_folder, name + ".nii.gz"))
        perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
        normalize_image(image_file, out_file, bias_correction=perform_bias_correction)
    # copy the truth file
    if truth_name is not None:
        try:
            truth_file = get_image(in_folder, truth_name)
        except RuntimeError:
            truth_file = get_image(in_folder, truth_name.split("_")[0])
        out_file = os.path.abspath(os.path.join(out_folder, "truth.nii.gz"))
        # shutil.copy(truth_file, out_file)
        convert_image_format(truth_file, out_file)
        check_origin(out_file, get_image(in_folder, ["t2"][0]))


def convert_tiantan_data(tiantan_folder, out_folder, no_bias_correction_modalities=("flair",), truth_name='tu'):
    """
    Preprocesses the BRATS data and writes it to a given output folder. Assumes the original folder structure.
    :param tiantan_folder: folder containing the original brats data
    :param out_folder: output folder to which the preprocessed data will be written
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
    for subject_folder in glob.glob(os.path.join(tiantan_folder, "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, subject)
            print(new_subject_folder)
            if os.path.exists(new_subject_folder):
                shutil.rmtree(new_subject_folder)
            os.makedirs(new_subject_folder)
            convert_tiantan_folder(subject_folder, new_subject_folder, truth_name,
                                   no_bias_correction_modalities=no_bias_correction_modalities)


def convert_truth_to_nii():
    """
    nibabel complains truth file is not nii
    :return:
    """
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", config["preprocessed"], "*", "*")):
        subject = (os.path.join(subject_dir, "truth.nii.gz"))
        training_data_files.append(subject)
    for file in training_data_files:
        file = os.path.join('/media/mingrui/960EVO/workspace/3DUnetCNN-fork/brats/', file)
        convert_image_format(file, file)
        print(file)


# doesn't work very well about 0.60
def fill_to_24_slices():
    data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", config["preprocessed"], "*", "*")):
        for subject in glob.glob(os.path.join(subject_dir, "*")):
            data_files.append(subject)
    for file in data_files:
        file = os.path.join('/media/mingrui/960EVO/workspace/3DUnetCNN-fork/brats/', file)
        img = nib.load(file)
        img_data = img.get_data()
        if img_data.shape[2] < 24:
            print("smaller than 24: " + file)
            layer_shape = img_data[:, :, 0].shape
            img_data = np.dstack((img_data, np.zeros(layer_shape)))
            new_img = nib.Nifti1Image(img_data, affine=img.affine)
            nib.save(new_img, file)
        elif img_data.shape[2] > 24:
            print("larger than 24: " + file)
            start = img_data.shape[2] - 24
            print(img_data[:, :, start:].shape)
            new_img = nib.Nifti1Image(img_data[:, :, start:], affine=img.affine)
            nib.save(new_img, file)


def convert_new_data():
    new_data_path = '/media/mingrui/960EVO/datasets/tiantan/new_data_201712/*/*/*'

    subject_list = []
    for subject_dir in glob.glob(new_data_path):
        t2_list = []
        for subject_file in glob.glob(os.path.join(subject_dir, '*')):
            if '/T2.nii' in subject_file:
                t2_list.append(subject_file)
            elif '/T2U.nii' in subject_file:
                t2_list.append(subject_file)
        if len(t2_list) == 2:
            subject_list.append(t2_list)
    print(len(subject_list))

    # make new folder
    new_folder_path = '/media/mingrui/960EVO/datasets/tiantan/new_data_201712/clean/'
    for i, t2_list in enumerate(subject_list):
        new_path = os.path.join(new_folder_path, str(i))
        os.mkdir(new_path)
        for t2_file in t2_list:
            if '/T2.nii' in t2_file:
                shutil.copyfile(t2_file, os.path.join(new_path, 't2.nii'))
            elif '/T2U.nii' in t2_file:
                shutil.copyfile(t2_file, os.path.join(new_path, 'tu.nii'))


def fill_to_512_x_512():
    '''
    Fill mri image width and height to 512 x 512.
    This means when training, don't need to reshape images into different shape
    :return:
    '''
    data_path = "/media/brainteam/hdd1/TiantanData/2017-11/tiantan_preprocessed_512"
    #data_path = "/mnt/960EVO/workspace/3DUnetCNN-fork/brats/data/tiantan_preprocessed_512_test"


    data_files = list()
    for subject_dir in glob.glob(os.path.join(data_path, "*", "*")):
        for subject in glob.glob(os.path.join(subject_dir, "*")):
            data_files.append(subject)
    for file in data_files:
        file = os.path.join(data_path, file)
        img = nib.load(file)
        img_data = img.get_data()
        # create new image of size 512 x 512
        new_size = (512, 512)
        # paste img_data onto center of new image
        new_data = []

        if img_data.shape[0] >= 512 and img_data.shape[1] >= 512:
            continue

        for d in range(img_data.shape[-1]):
            old_slice = img_data[..., d]
            new_slice = np.zeros(new_size)
            new_slice[(new_size[0] - old_slice.shape[0]) / 2: (new_size[0] + old_slice.shape[0]) / 2,
            (new_size[1] - old_slice.shape[1]) / 2: (new_size[1] + old_slice.shape[1]) / 2] = old_slice
            new_data.append(new_slice)

        new_data_np = np.dstack(new_data)
        print(new_data_np.shape)
        new_img = nib.Nifti1Image(new_data_np, affine=img.affine)
        new_file_path = file
        if os.path.exists(new_file_path):
            os.remove(new_file_path)
        nib.save(new_img, new_file_path)


parser = argparse.ArgumentParser(description='tiantan data preprocessing')

parser.add_argument('--input-path', type=str, default=None, metavar='str',
                    help='original data path')
parser.add_argument('--output-path', type=str, default=None, metavar='str',
                    help='preprocessed data output path')

args = parser.parse_args()

if __name__ == "__main__":
    print('tiantan preprocess')
    # convert_truth_to_nii()
    # fill_to_24_slices()
    # convert_new_data()
    # convert_tiantan_data('/media/mingrui/960EVO/datasets/tiantan/2017-11/training_tiantan_IDH1', '/media/mingrui/960EVO/datasets/tiantan/2017-11/tiantan_preprocessed/')
    # convert_tiantan_data('/media/mingrui/960EVO/datasets/tiantan/2017-11/valid_tiantan_IDH1', '/media/mingrui/960EVO/datasets/tiantan/2017-11/tiantan_preprocessed/')
    # convert_tiantan_data('/media/mingrui/960EVO/datasets/tiantan/2017-12/clean', '/media/mingrui/960EVO/datasets/tiantan/2017-12//tiantan_preprocessed/')

    # fill_to_512_x_512()

    convert_tiantan_data(args.input_path,
                         args.output_path,
                         truth_name=None)
