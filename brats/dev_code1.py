from keras.models import Model
from keras.layers import Flatten, Conv3D, MaxPooling3D, Activation, Dense, BatchNormalization
from unet3d.training import load_old_model
from keras import optimizers
import os
from keras.engine import Input, Model
from keras.optimizers import Adam
from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.training import load_old_model, train_model

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

config["data_file"] = os.path.abspath("brats_data.h5")
config["model_file"] = os.path.abspath("tumor_segmentation_model_tiantan.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.

config["preprocessed"] = "tiantan_only" # change this to use different data files



def classification_model(model_file):
    old_model = load_old_model(model_file)
    output_layer = old_model.layers[11]
    #print output_layer.output
    print("CLASSIFICATION MODEL WITH TRANSFER")
    x = output_layer.output
    print(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name='max_pooling3d_4')(x)

    x = Conv3D(512, (3, 3, 3),  padding='same', name='conv3d_9')(x)
    x = BatchNormalization(axis=1, name='batch_normalization_4')(x)
    x = Activation('relu', name='activation_5')(x)

    # x = Conv3D(512, (3, 3, 3),  padding='same', name='conv3d_10')(x)
    # x = BatchNormalization(axis=1, name='batch_normalization_5')(x)
    # x = Activation('relu', name='activation_6')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2), name='max_pooling3d_5', padding='same')(x)


    x = Conv3D(1024, (3, 3, 3), padding='same', name='conv3d_11')(x)
    x = BatchNormalization(axis=1, name='batch_normalization_6')(x)
    x = Activation('relu', name='activation_7')(x)

    # x = Conv3D(1024, (3, 3, 3), padding='same', name='conv3d_12')(x)
    # x = BatchNormalization(axis=1, name='batch_normalization_7')(x)
    # x = Activation('relu', name='activation_8')(x)

    x = MaxPooling3D(pool_size=(2, 2, 2), name='max_pooling3d_6')(x)


    # x = Conv3D(2048, (3, 3, 3), padding='same', name='conv3d_13')(x)
    # x = BatchNormalization(axis=1, name='batch_normalization_8')(x)
    # x = Activation('relu', name='activation_9')(x)

    x = Conv3D(2048, (3, 3, 3), padding='same', name='conv3d_14')(x)
    x = BatchNormalization(axis=1, name='batch_normalization_9')(x)
    x = Activation('relu',name='activation_10')(x)
    #x = MaxPooling3D(pool_size=(2, 2, 2), name='max_pooling3d_7')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model_final = Model(input= old_model.input, output=predictions)
    #model_final.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001, momentum=0.8, decay=0.8),metrics=['accuracy'])
    print('Adam Optimizer')
    for layer in model_final.layers[:12]:
        print(layer.output)
        layer.trainable = False

    model_final.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001),
                        metrics=['accuracy'])
    #print model_final

    # for layer in model_final.layers:
    #     print(layer.output)
    return model_final


def classification_model_2(input_shape, pool_size=(2, 2, 2),
                  depth=4,n_base_filters=32,
                  batch_normalization=True):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    print("CLASSIFICATION MODEL 2")
    inputs = Input(input_shape)
    current_layer = inputs
    #levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            #levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            #levels.append([layer1, layer2])
    current_layer = Flatten()(current_layer)
    current_layer = Dense(512, activation='relu')(current_layer)
    current_layer = Dense(128, activation='relu')(current_layer)
    current_layer = Dense(32, activation='relu')(current_layer)
    predictions = Dense(2, activation='softmax')(current_layer)
    model_final = Model(input=inputs, output=predictions)
    model_final.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001),
                        metrics=['accuracy'])

    #final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    #act = Activation('sigmoid')(final_convolution)
    #model = Model(inputs=inputs, outputs=act)

    # if not isinstance(metrics, list):
    #     metrics = [metrics]
    #
    # if include_label_wise_dice_coefficients and n_labels > 1:
    #     label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    #     if metrics:
    #         metrics = metrics + label_wise_dice_metrics
    #     else:
    #         metrics = label_wise_dice_metrics
    #
    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=metrics)
    return model_final

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same'):
    """

    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def main(overwrite=False):
    # convert input images into an hdf5 file
    # if overwrite or not os.path.exists(config["data_file"]):
    #     training_files = fetch_training_data_files()
    #
    #     write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])
    data_file_opened = open_data_file(config["data_file"])
    #
    # if not overwrite and os.path.exists(config["model_file"]):
    #     model = load_old_model(config["model_file"])
    # else:
    #     # instantiate new model
    #     model = unet_model_3d(input_shape=config["input_shape"],
    #                           pool_size=config["pool_size"],
    #                           n_labels=config["n_labels"],
    #                           initial_learning_rate=config["initial_learning_rate"],
    #                           deconvolution=config["deconvolution"])
    model = classification_model('tumor_segmentation_model_tiantan_no_patch_bn.h5')#classification_model_2(input_shape=config["input_shape"])
    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()



if __name__ == "__main__":
    main()
    #model = classification_model('tumor_segmentation_model_tiantan_no_patch.h5')