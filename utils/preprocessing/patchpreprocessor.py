__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, width, height):
        # store the target width and height of the image
        self.width = width
        self.height = height

    def preprocess(self, image):
        # extract a random crop from the image with the target width
        # and height
        return extract_patches_2d(image, (self.height, self.width),
                                  max_patches=1)[0]
