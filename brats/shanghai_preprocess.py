from pydcmio.dcmconverter.converter import generate_config
from pydcmio.dcmconverter.converter import dcm2nii
from shutil import copyfile

import glob
import os
import warnings
import shutil

import SimpleITK as sitk
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

from train import config

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
            if name == truth_name and not (image.GetOrigin() == background_image.GetOrigin()):
                image.SetOrigin(background_image.GetOrigin())
            background_image = sitk.And(image == 0, background_image)
        else:
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
    file_card = os.path.join(subject_folder, "*" + name + "*.nii.gz")
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
        print('bias correction true')
        correct_bias(in_file, out_file)
    else:
        print('bias correction false')
        shutil.copy(in_file, out_file)
    return out_file

def put_shanghai_dataset_in_single_folders():
    shanghai_folder = '/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/dcm/*'
    shanghai_single_folder = '/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/dcm_single_folders'

    for subject_folder in glob.glob(shanghai_folder):
        data_dir = ''
        if  'data-110' in subject_folder:
            data_dir = 'data-110'
        elif 'data-119' in subject_folder:
            data_dir = 'data-119'
        output_folder = os.path.join(shanghai_single_folder, data_dir)

        for subject in glob.glob(os.path.join(subject_folder, '*')):
            file_name = os.path.basename(subject)
            file_dir = os.path.join(output_folder, os.path.splitext(file_name)[0])
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            new_name = ''
            if 'Flair' in subject:
                new_name = 'flair.nii.gz'
            elif 'T1' in subject:
                new_name = 't1.nii.gz'
            print(subject)
            copyfile(subject, os.path.join(file_dir, new_name))


def dcm2nii_shanghai_dataset():
    shanghai_single_folder = '/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/dcm_single_folders/*/*'
    shanghai_output_folder = '/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/nifti/'

    '''niidir = '/media/mingrui/960EVO/datasets/tiantan/shanghai/test/*'
    test_folder = '/media/mingrui/960EVO/datasets/tiantan/shanghai/test'
    for subject_folder in glob.glob(niidir):
        config_file = generate_config(
            test_folder, anonymized=False, gzip=True, add_date=False,
            add_acquisition_number=False, add_protocol_name=False,
            add_patient_name=False, add_source_filename=True,
            begin_clip=0, end_clip=0)


        (files, reoriented_files, reoriented_and_cropped_files,
         bvecs, bvals) = dcm2nii(
            subject_folder, o=test_folder, b=config_file)'''

    for subject_folder in glob.glob(shanghai_single_folder):
        output_path = ''
        if 'data-110' in subject_folder:
            output_path = os.path.join(shanghai_output_folder, 'data-110', os.path.basename(subject_folder))
        elif 'data-119' in subject_folder:
            output_path = os.path.join(shanghai_output_folder, 'data-119', os.path.basename(subject_folder))

        print(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        config_file = generate_config(
            output_path, anonymized=False, gzip=True, add_date=False,
            add_acquisition_number=False, add_protocol_name=False,
            add_patient_name=False, add_source_filename=True,
            begin_clip=0, end_clip=0)
        (files, reoriented_files, reoriented_and_cropped_files,
         bvecs, bvals) = dcm2nii(
            subject_folder, o=output_path, b=config_file)

def convert_shanghai_folder(in_folder, out_folder, truth_name="tu",
                         no_bias_correction_modalities=None):
    for name in ["flair"]:
        image_file = get_image(in_folder, name)
        out_file = os.path.abspath(os.path.join(out_folder, name + ".nii.gz"))
        perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
        normalize_image(image_file, out_file, bias_correction=perform_bias_correction)
    # copy the truth file
    #try:
    #    truth_file = get_image(in_folder, truth_name)
    #except RuntimeError:
    #    truth_file = get_image(in_folder, truth_name.split("_")[0])
    #out_file = os.path.abspath(os.path.join(out_folder, "truth.nii.gz"))
    #shutil.copy(truth_file, out_file)
    #convert_image_format(truth_file, out_file)
    check_origin(out_file, get_image(in_folder, ["flair"][0]))

def convert_shanghai_t1_folder(in_folder, out_folder, truth_name="tu",
                            no_bias_correction_modalities=None):
    for name in ["t1"]:
        image_file = get_image(in_folder, name)
        out_file = os.path.abspath(os.path.join(out_folder, name + ".nii.gz"))
        perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
        normalize_image(image_file, out_file, bias_correction=perform_bias_correction)
    # copy the truth file
    # try:
    #    truth_file = get_image(in_folder, truth_name)
    # except RuntimeError:
    #    truth_file = get_image(in_folder, truth_name.split("_")[0])
    # out_file = os.path.abspath(os.path.join(out_folder, "truth.nii.gz"))
    # shutil.copy(truth_file, out_file)
    # convert_image_format(truth_file, out_file)
    check_origin(out_file, get_image(in_folder, ["t1"][0]))

def convert_shanghai_data(data_folder, out_folder, overwrite=True, no_bias_correction_modalities =('None',)):
    """
    Preprocesses the BRATS data and writes it to a given output folder. Assumes the original folder structure.
    :param data_folder: folder containing the original brats data
    :param out_folder: output folder to which the preprocessed data will be written
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
    for subject_folder in glob.glob(os.path.join(data_folder, "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                convert_shanghai_folder(subject_folder, new_subject_folder,
                                     no_bias_correction_modalities=no_bias_correction_modalities)

def convert_shanghai_t1_data(data_folder, out_folder, overwrite=True, no_bias_correction_modalities =('None',)):
    """
    Preprocesses the BRATS data and writes it to a given output folder. Assumes the original folder structure.
    :param data_folder: folder containing the original brats data
    :param out_folder: output folder to which the preprocessed data will be written
    :param overwrite: set to True in order to redo all the preprocessing
    :param no_bias_correction_modalities: performing bias correction could reduce the signal of certain modalities. If
    concerned about a reduction in signal for a specific modality, specify by including the given modality in a list
    or tuple.
    :return:
    """
    for subject_folder in glob.glob(os.path.join(data_folder, "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                convert_shanghai_t1_folder(subject_folder, new_subject_folder,
                                     no_bias_correction_modalities=no_bias_correction_modalities)

def convert_dcm_2_nii(dcm_folder):
	for subject_folder in glob.glob(dcm_folder):
		print(subject_folder)
		output_path = subject_folder
		config_file = generate_config(
            output_path, anonymized=False, gzip=True, add_date=False,
            add_acquisition_number=False, add_protocol_name=False,
            add_patient_name=False, add_source_filename=True,
            begin_clip=0, end_clip=0)
        (files, reoriented_files, reoriented_and_cropped_files,
         bvecs, bvals) = dcm2nii(
            subject_folder, o=output_path, b=config_file)

if __name__ == "__main__":
    print('shanghai preprocess')
    #put_shanghai_dataset_in_single_folders()
    #dcm2nii_shanghai_dataset()
    #convert_shanghai_data('/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/nifti/data-110', '/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/shanghai_preprocessed/')
    #convert_shanghai_data('/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/nifti/data-119', '/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/shanghai_preprocessed/')
    #convert_shanghai_t1_data('/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/nifti/data-119-t1', '/media/mingrui/960EVO/datasets/tiantan/2017-9-shanghai/shanghai_preprocessed/')
    convert_dcm_2_nii('/media/mingrui/DATA/UbuntuDownload/HuyeDcm2Nii/dcm/*')

