__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# USAGE
# python style_transfer.py

import os

import tensorflow as tf
from pyimagesearch.nn.conv.neuralstyle import NeuralStyle

# import necessary packages
from config import style_transfer_config as config


def loadImage(imagePath):
    # specify the maximum dimension to which the image is to be
    # resized
    maxDim = 512

    # load the image from the given path, convert the image bytes
    # to a tensor, and convert the data type of the image
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # grab the height and width of the image, cast them to floats,
    # determine the larger dimension between height and width, and
    # determine the scaling factor
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = maxDim / long_dim

    # scale back the new shape, cast it to an integer, resize the
    # image to the new shape, and  add a batch dimension
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]

    # return the resized image
    return image


@tf.function
def train_one_step(image, style_targets, content_targets):
    # derive the style and content loss weight values
    style_weight = config.style_weight / len(config.style_layers)
    content_weight = config.content_weight / len(config.content_layers)

    # keep track of our gradients
    with tf.GradientTape() as tape:
        # run the content image through our neural style network to
        # get its features, determine the loss, and add total
        # variational loss to regularize it
        outputs = extractor(image)
        loss = extractor.styleContentLoss(outputs, style_targets,
                                          content_targets, style_weight, content_weight)
        loss += config.tvWeight * tf.image.total_variation(image)

    # grab the gradients of the loss with respect to the image and
    #  apply the gradients to update the image after clipping the
    # values to [0, 1] range
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(extractor.clipPixels(image))


# initialize the Adam optimizer
opt = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.99,
                         epsilon=1e-1)

# load the content and style images
print("[INFO] loading content and style images...")
content_image = loadImage(config.content_image)
style_image = loadImage(config.style_image)

# grab the contents layer from which feature maps will be extracted
# along with the style layer blocks
content_layers = config.content_layers
style_layers = config.style_layers

# initialize the our network to extract features from the style and
# content images
print("[INFO] initializing off the extractor network...")
extractor = NeuralStyle(style_layers, content_layers)

# extract the features from the style and content images
style_targets = extractor(style_image)["style"]
content_targets = extractor(content_image)["content"]

# initialize the content image as a TensorFlow variable along with
# the total number of steps taken in the current epoch
print("[INFO] training the style transfer model...")
image = tf.Variable(content_image)
step = 0

# loop over the number of epochs
for epoch in range(config.epochs):
    # loop over the number of steps in the epoch
    for i in range(config.steps_per_epoch):
        # perform a single training step, then increment our step
        # counter
        train_one_step(image, style_targets, content_targets)
        step += 1

    # construct the path to the intermediate resulting image (for
    # visualization purposes) and save it
    print("[INFO] training step: {}".format(step))
    p = "_".join([str(epoch), str(i)])
    p = "{}.png".format(p)
    p = os.path.join(config.interm_outputs, p)
    extractor.tensorToImage(image).save(p)

# save the final stylized image
extractor.tensorToImage(image).save(config.final_image)
