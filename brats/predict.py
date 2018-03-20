import os

from brats.train import config
from unet3d.prediction import run_validation_cases

from unet3d.prediction import patch_wise_prediction
from unet3d.prediction import prediction_to_image
from unet3d.training import load_old_model

import nibabel as nib
import numpy as np

def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)

def make_prediction():
    print("make prediction: ")
    test_file = "/mnt/960EVO/workspace/3DUnetCNN-fork/brats/prediction/validation_case_0/data_t2.nii.gz"
    out_file = "/mnt/960EVO/workspace/3DUnetCNN-fork/brats/prediction/validation_case_0/test_prediction.nii.gz"
    model_file = "/mnt/960EVO/workspace/3DUnetCNN-fork/brats/tumor_segmentation_model.h5"
    model_file = config["model_file"]
    model = load_old_model(model_file)

    img = nib.load(test_file)
    img_data = img.get_data()
    img_data = np.array([img_data])
    img_data = np.array([img_data])

    prediction = patch_wise_prediction(model=model, data=img_data, overlap=16)
    #print(type(prediction))
    #new_img = nib.Nifti1Image(prediction, affine=np.eye(4))
    #nib.save(new_img, out_file)
    prediction_image = prediction_to_image(prediction, affine=img.affine, label_map=True, threshold=0.7)
    prediction_image.to_filename(out_file)

if __name__ == "__main__":
    make_prediction()
    #ain()
