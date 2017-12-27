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
from config import *


def classification_model(model_file):
    print("CLASSIFICATION MODEL WITH TRANSFER")
    old_model = load_old_model(model_file)
    output_layer = old_model.layers[11]

    x = output_layer.output

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
    x = Dropout(x)

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
    for layer in model_final.layers[:11]:
        print(layer.output)
        layer.trainable = False

    model_final.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config["initial_learning_rate"]),
                        metrics=['accuracy'])

    return model_final


def classification_model_2(input_shape, pool_size=(2, 2, 2),
                  depth=4,n_base_filters=32,
                  batch_normalization=True):

    print("CLASSIFICATION MODEL DIRECT TRAINING")
    inputs = Input(input_shape)
    current_layer = inputs

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
        else:
            current_layer = layer2
    current_layer = Flatten()(current_layer)
    current_layer = Dense(512, activation='relu')(current_layer)
    current_layer = Dense(128, activation='relu')(current_layer)
    current_layer = Dense(32, activation='relu')(current_layer)
    predictions = Dense(2, activation='softmax')(current_layer)
    model_final = Model(input=inputs, output=predictions)
    model_final.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config["initial_learning_rate"]),
                        metrics=['accuracy'])

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

    data_file_opened = open_data_file(config["data_file"])

    model = classification_model('tumor_segmentation_model_tiantan_no_patch_include201712_bn_87_74_20171227.h5')
    #('tumor_segmentation_model_tiantan_no_patch_bn.h5')#classification_model_2(input_shape=config["input_shape"])
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