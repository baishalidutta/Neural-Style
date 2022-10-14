__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

import h5py
import numpy as np
# import the necessary packages
from tensorflow.keras.utils import to_categorical


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size, preprocessors=None,
                 aug=None, binarize=True, classes=2):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(db_path, "r")
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batch_size):
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i: i + self.batch_size]
                labels = self.db["labels"][i: i + self.batch_size]

                # check to see if the labels should be binarized
                if self.binarize:
                    labels = to_categorical(labels,
                                            self.classes)

                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    proc_images = []

                    # loop over the images
                    for image in images:
                        # loop over the preprocessors and apply each
                        # to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update the list of processed images
                        proc_images.append(image)

                    # update the images array to be the processed
                    # images
                    images = np.array(proc_images)

                # if the data augmentor exists, apply it
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images,
                                                          labels, batch_size=self.batch_size))

                # yield a tuple of images and labels
                yield (images, labels)

            # increment the total number of epochs
            epochs += 1

    def close(self):
        # close the database
        self.db.close()
