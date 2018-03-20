# setting for classification

import os

#machine = 'brainteam'
machine = 'mingrui'

config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape"] = (128, 128, 24)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None #(128, 128, 24)#None#(16, 16, 62)  # switch to None to train on the whole image
config["labels"] = (1, )  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1Gd", "flair", "t2"]
config["tiantan_modalities"] = ["t2"]
config["training_modalities"] = config["tiantan_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = False  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 2
config["validation_batch_size"] = 6
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 5  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 100  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("tiantan_data_classification.h5")
config["model_file"] = os.path.abspath("tumor_segmentation_model_tiantan_classification.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.

config["preprocessed"] = "tiantan_only" # change this to use different data files

config["segmentation_mode"] = True # True for segmentation, False for classification




'''
Below is config for segmentation
'''

config_one = dict()
config_one["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config_one["image_shape"] = (512, 512, 24)#(256, 256, 24)  # This determines what shape the images will be cropped/resampled to.
config_one["patch_shape"] = (64, 64, 24)  # switch to None to train on the whole image
config_one["labels"] = (1, )  # the label numbers on the input image
config_one["n_labels"] = len(config["labels"])
config_one["all_modalities"] = ["t1", "t1Gd", "flair", "t2"]
config_one["tiantan_modalities"] = ["t2"]
config_one["training_modalities"] = config_one["tiantan_modalities"]  # change this if you want to only use some of the modalities
config_one["nb_channels"] = len(config_one["training_modalities"])
if "patch_shape" in config_one and config_one["patch_shape"] is not None:
    config_one["input_shape"] = tuple([config_one["nb_channels"]] + list(config_one["patch_shape"]))
else:
    config_one["input_shape"] = tuple([config_one["nb_channels"]] + list(config_one["image_shape"]))
config_one["truth_channel"] = config_one["nb_channels"]
config_one["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config_one["batch_size"] = 4
config_one["validation_batch_size"] = 6
config_one["n_epochs"] = 500  # cutoff the training after this many epochs
config_one["patience"] = 30  # learning rate will be reduced after this many epochs if the validation loss is not improving
config_one["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config_one["initial_learning_rate"] = 0.0001
config_one["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config_one["validation_split"] = 0.8  # portion of the data that will be used for training
config_one["flip"] = False  # augments the data by randomly flipping an axis during
config_one["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config_one["distort"] =  None # switch to None if you want no distortion
config_one["augment"] = config_one["flip"] or config_one["distort"]
config_one["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config_one["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config_one["skip_blank"] = True  # if True, then patches without any target will be skipped

#config["data_file"] = os.path.abspath("tiantan_data_classification.h5")
config_one["data_file"] = os.path.abspath("brats_data.h5")
config_one["model_file"] = os.path.abspath("tumor_segmentation_model.h5")
#config["model_file"] = os.path.abspath("/media/mingrui/960EVO/workspace/3DUnetCNN-fork/brats/20171201/tumor_segmentation_model.h5")
config_one["training_file"] = os.path.abspath("training_ids.pkl")
config_one["validation_file"] = os.path.abspath("validation_ids.pkl")
config_one["overwrite"] = True # If True, will previous files. If False, will use previously written files.

#config["preprocessed"] = "tiantan_preprocessed" # tiantan november first batch data
config_one["preprocessed"] = "tiantan_preprocessed_512" # test data
#config["preprocessed"] = "tiantan_skull_strip" # change this to use different data files
if machine == 'brainteam':
    config_one["preprocessed"] = "/media/brainteam/hdd1/TiantanData/2017-11/tiantan_preprocessed" # change this to use different data files
#config["preprocessed"] = "tiantan_201712_preprocessed" # change this to use different data files
#config["preprocessed"] = "brats_2017_preprocessed" # brats data
#config["preprocessed"] = "201712_new" # tiantan new december 2017 data
