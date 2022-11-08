​# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================

#
# Computer Vision with CNNs
# For this exercise you will use the beans dataset from TFDS
# to build a classifier that recognizes different types of bean disease
# Please make sure you keep the given layers as shown, or your submission
# will fail to be graded. Please also note the image size of 224x224



import tensorflow as tf
import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import tensorflow_datasets as tfds
import warnings

warnings.filterwarnings('ignore')

# def map_data(image, label, target_height = 224, target_width = 224):
#     """Normalizes images: `unit8` -> `float32` and resizes images
#     by keeping the aspect ratio the same without distortion."""
#     image = # Your Code here to normalize the image
#     image = tf.image.resize_with_crop_or_pad(# Parameters to resize and crop the image as desired)
#     return # Return the appropriate parameters

def solution_model():
    # (ds_train, ds_validation, ds_test), ds_info = tfds.load(
    datasets, info = tfds.load(name = 'beans', with_info = True, as_supervised = True, split = ['train', 'test', 'validation'])
    train, info_train = tfds.load(name = 'beans', with_info = True, split = 'test')


    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return tf.image.resize(image, [224, 224]), tf.one_hot(label, 3)

    def get_dataset(batch_size = 32):
        train_dataset_sclaed = datasets[0].map(scale).shuffle(1000).batch(batch_size)
        test_dataset_sclaed = datasets[1].map(scale).batch(batch_size)

        validation_dataset_sclaed = datasets[2].map(scale).batch(batch_size)


        return train_dataset_sclaed, test_dataset_sclaed, validation_dataset_sclaed


train_dataset, test_dataset, val_dataset = get_dataset()
train_dataset.cache()
val_dataset.cache()
feature_extractor = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
feature_extractor_layer = hub.KerasLayer(feature_extractor, input_shape = (224, 224, 3))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential(

    [
    feature_extractor_layer,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation = 'softmax')
    ]

    )

model.compile(
optimizer = tf.keras.optimizers.Adam(),
loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
metrics = ['acc']
)

model.fit(train_dataset, epochs = 6, validation_data = val_dataset)
        # name=#Dataset sames,
        # split=[#Desired Splits],
        # as_supervised=#Appropriate parameter,
        # with_info=#Appropriate parameter)

#     ds_train = # Perform appropriate operations to prepare ds_train
#     ds_validation = # Perform appropriate operations to prepare ds_validation
#     ds_test = # Perform appropriate operations to prepare ds_test
#     model = tf.keras.models.Sequential([
#       # You can change any parameters here *except* input_shape
#       tf.keras.layers.Conv2D(16, (3, 3), input_shape=(224, 224, 3), strides=2, padding='same', activation = 'relu'),
#       # Add whatever layers you like
#       # Keep this final layer UNCHANGED
#       tf.keras.layers.Dense(3, activation='softmax'),
#     ])


#     model.compile(
#         # Choose appropriate parameters
#     )


#     history = model.fit(
#         # Choose appropriate parameters
#     )




    return model


if __name__ == '__main__':

    model = solution_model()

    model.save("c3q4.h5")

