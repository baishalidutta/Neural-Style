__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

import imutils
from tensorflow.keras.applications import imagenet_utils


def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield x, y, image[y:y + ws[1], x:x + ws[0]]


def image_pyramid(image, scale=1.5, min_size=(224, 224)):
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield image


def classify_batch(model, batch_rois, batch_locs, labels, min_prob=0.5,
                   top=10, dims=(224, 224)):
    # pass our batch ROIs through our network and decode the
    # predictions
    preds = model.predict(batch_rois)
    P = imagenet_utils.decode_predictions(preds, top=top)

    # loop over the decoded predictions
    for i in range(0, len(P)):
        for (_, label, prob) in P[i]:
            # filter out weak detections by ensuring the
            # predicted probability is greater than the minimum
            # probability
            if prob > min_prob:
                # grab the coordinates of the sliding window for
                # the prediction and construct the bounding box
                (pX, pY) = batch_locs[i]
                box = (pX, pY, pX + dims[0], pY + dims[1])

                # grab the list of predictions for the label and
                # add the bounding box + probability to the list
                L = labels.get(label, [])
                L.append((box, prob))
                labels[label] = L

    # return the labels dictionary
    return labels
