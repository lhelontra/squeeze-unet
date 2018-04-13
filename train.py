#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from keras.layers import Input
from keras.optimizers import Adam

import cvision
from nnutil.dataset import ImageDataGenerator
from squeezeunet import SqueezeUNet

img_rows = 224
img_cols = 224
channels = 3
epochs = 10
batch_size = 1
nb_train_samples = 5000
nb_validation_samples = 2000
save_to_dir = None
inputs = Input((img_rows, img_cols, channels))

class_indices = {
          "unlabelled" : 0,
          "sky" : 1,
          "building" : 2,
          "pole" : 3,
          "road" : 4,
          "pavement" : 5,
          "tree" : 6,
          "signsymbol" : 7,
          "fence": 8,
          "car" : 9,
          "pedestrian" : 10,
          "bicyclist" : 11
}

datasets = ["~/datasets/camvid/train-camvid.json"]
validation_datasets = ["~/datasets/camvid/validation-camvid.json"]


pipeline_train_transform = cvision.importFromSketch("sketchs/train_pipeline.sketch")
train_imGenerator = ImageDataGenerator(pipeline_train_transform,
                                        dataset_mean=None,
                                        dataset_std_normalization=None)


segmentation_generator = train_imGenerator.flow_json_segmentation(datasets, class_indices=class_indices,
                                                                    class_mode="categorical",
                                                                    skipImageNonAnnotations=True,
                                                                    nonAnnotationLabel=None,
                                                                    batch_size=batch_size,
                                                                    mask_rescale=None,
                                                                    mask_transform=None,
                                                                    save_to_dir=save_to_dir,
                                                                    save_prefix='test',
                                                                    save_format='jpg')

validation_generator = train_imGenerator.flow_json_segmentation(validation_datasets, class_indices=class_indices,
                                                                    class_mode="categorical",
                                                                    skipImageNonAnnotations=True,
                                                                    nonAnnotationLabel=None,
                                                                    batch_size=batch_size,
                                                                    mask_rescale=None,
                                                                    mask_transform=None,
                                                                    save_to_dir=save_to_dir,
                                                                    save_prefix='test',
                                                                    save_format='jpg')

model = SqueezeUNet(inputs, num_classes=12, deconv_ksize=3, activation='softmax')
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-05), metrics=["accuracy"])
if os.path.exists('squeezeunet.h5'):
    model.load_weights('squeezeunet.h5')

model.fit_generator(segmentation_generator,
                    class_weight="auto",
                    steps_per_epoch=nb_train_samples,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples)

model.save_weights('squeezeunet.h5', overwrite=True)
