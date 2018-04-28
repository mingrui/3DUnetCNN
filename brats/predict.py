import os

from unet3d.prediction import run_validation_cases

from unet3d.prediction import patch_wise_prediction
from unet3d.prediction import prediction_to_image
from unet3d.training import load_old_model

import nibabel as nib
import numpy as np

from config import config_isensee
config = config_isensee

def main(prediction_dir):
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file='isensee_2017_model_freeze.h5',
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,
                         threshold=0.5)

def make_prediction():
    print("make prediction: ")
    test_file = "/mnt/960EVO/workspace/3DUnetCNN-fork/brats/prediction/pred/t2.nii.gz"
    out_file = "/mnt/960EVO/workspace/3DUnetCNN-fork/brats/prediction/pred/pred.nii.gz"
    model_file = "/mnt/960EVO/workspace/3DUnetCNN-fork/brats/isensee_2017_model.h5"
    model = load_old_model(model_file)

    img = nib.load(test_file)
    img_data = img.get_data()
    img_data = np.array([img_data])
    img_data = np.array([img_data])

    prediction = patch_wise_prediction(model=model, data=img_data, overlap=16)[np.newaxis]
    #print(type(prediction))
    #new_img = nib.Nifti1Image(prediction, affine=np.eye(4))
    #nib.save(new_img, out_file)
    prediction_image = prediction_to_image(prediction, affine=img.affine, label_map=True, threshold=0.7)
    prediction_image.to_filename(out_file)


if __name__ == "__main__":
    # make_prediction()
    prediction_dir = '/mnt/960EVO/datasets/tiantan/2017-11/tiantan_preprocessed_png/512/0_3dunet_nifti_layer_prediction'
    main(prediction_dir)
