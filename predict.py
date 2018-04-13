#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from keras.layers import Input
from nnutil.image import load_img, im_standardize

import cvision
from squeezeunet import SqueezeUNet

def label_img(img):
    img = img.copy()
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    #Road_marking = [255,69,0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]
    Unlabelled = [0, 0, 0]

    label_colours = [Unlabelled,
                        Sky,
                        Building,
                        Pole,
                        Road,
                        Pavement,
                        Tree,
                        SignSymbol,
                        Fence,
                        Car,
                        Pedestrian,
                        Bicyclist]

    categorical_to_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    if img.ndim < 3 or img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for l in range(len(categorical_to_id)):
        img[np.where((img == [l, l, l]).all(axis=-1))] = label_colours[l]
    return img


img_rows = 224
img_cols = 224
channels = 3
inputs = Input((img_rows, img_cols, channels))

model = SqueezeUNet(inputs, num_classes=12, deconv_ksize=3, activation='softmax')
model.load_weights('squeezeunet.h5')

pipeline_predict_transform = cvision.importFromSketch("sketchs/predict_pipeline.sketch")

img = load_img("~/datasets/camvid/images_validation/0001TP_006720.png", color_mode="bgr")
img = pipeline_predict_transform.run(img)
img = im_standardize(img, rescale=None, mean=None, std=None)

masks = model.predict(np.expand_dims(img, axis=0), verbose=0)
mask = np.argmax(masks[0], axis=2).astype(np.uint8)
mask = label_img(mask)

cv2.imshow("segmentation", mask)
cv2.waitKey(0)
