import pickle
import os
import collections
import random

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like, resample_img

from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """
    
    :param image_files: 
    :param image_shape: 
    :param crop: 
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return: 
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list


def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    print('image.shape : ', image.shape, 'target shape : ', image_shape)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        debug_name = ''.join(in_file.split('/')[-2:])
        return resize_new(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize_new(image, new_shape, interpolation="linear", debug_name=None):
    # image = reorder_img(image, resample=interpolation)
    zoom_level = np.true_divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    # np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    # new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())


    new_img = new_img_like(image, new_data, affine=new_affine)

    if debug_name is not None:
        print('new image shape : ', new_img.shape)
        nib.save(new_img, debug_name)

    return new_img

def resize(image, new_shape, interpolation='linear'):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    #print("input_shape: {}".format(input_shape))
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    #print("output_shape: {}".format(output_shape))
    new_spacing = input_shape/output_shape
    #print("new_spacing: {}".format(new_spacing))
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)
