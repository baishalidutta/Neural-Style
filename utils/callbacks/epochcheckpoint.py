__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

import os

# import the necessary packages
from tensorflow.keras.callbacks import Callback


class EpochCheckpoint(Callback):
    def __init__(self, output_path, every=5, start_at=0):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.int_epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output_path,
                                  "epoch_{}.hdf5".format(self.int_epoch + 1)])
            self.model.save(p, overwrite=True)

        # increment the internal epoch counter
        self.int_epoch += 1
