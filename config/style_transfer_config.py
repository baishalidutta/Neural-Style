__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# import the necessary packages
import os

# define the content layer from which feature maps will be extracted
content_layers = ["block4_conv2"]

# define the list of style layer blocks from our pre-trained CNN
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]

# define the style weight, content weight, and total-variational
# loss weight, respectively (these are the values you'll want to
# tune to generate new style transfers)
style_weight = 1.0
content_weight = 1e4
tv_weight = 20.0

# define the number of epochs to train for along with the steps
# per each epoch
epochs = 15
steps_per_epoch = 100

# define the path to the input content image, input style image,
# final output image, and path to the directory that will store
# the intermediate outptus
content_image = os.path.sep.join(["inputs", "jp.jpg"])
style_image = os.path.sep.join(["inputs", "mcescher.jpg"])
final_image = "final.png"
interm_outputs = "intermediate_outputs"
