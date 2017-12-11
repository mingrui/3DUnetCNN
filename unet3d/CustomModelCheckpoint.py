import keras
import numpy as np

class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, model, path):

        super(CustomModelCheckpoint, self).__init__()

        self.model = model
        self.path = path

        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):

        loss = logs['val_loss']

        if loss < self.best_loss:

            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = loss