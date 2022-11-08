# ======================================================================
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
#
# For this exercise, build and train a cats v dogs classifier
# using the Cats v Dogs dataset from TFDS.
# Be sure to use the final layer as shown 
#     (Dense, 2 neurons, softmax activation)
#
# The testing infrastructure will resize all images to 224x224
# with 3 bytes of color depth. Make sure your input layer trains
# images to that specification, or the tests will fail.
#
# Make sure your output layer is exactly as specified here, or the 
# tests will fail.
#
# HINT: This is a large dataset and might take a long time to train.
# When experimenting with your architecture, use the splits API to train
# on a smaller set, and then gradually increase the training set size until
# it works very well. This is trainable in reasonable time, even on a CPU
# if architected correctly.
# NOTE: The dataset has some corrupt JPEG data in the images. If you see warnings
# about extraneous bytes before marker 0xd9, you can ignore them safely

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def preprocess(features):
    # YOUR CODE HERE
    image, label = tf.cast(features['image'], tf.float32) / 255.0, tf.one_hot(features['label'], 2)
    image = tf.image.resize(image, size=(224, 224))
    return image, label

dataset_name = 'cats_vs_dogs'
train_dataset, info = tfds.load(name=dataset_name, split="train[:20000]", with_info=True)
valid_dataset = tfds.load(name=dataset_name, split="train[20000:]")

train_dataset = train_dataset.repeat().map(preprocess).batch(32)
valid_dataset = valid_dataset.repeat().map(preprocess).batch(32)

total_size = 20000
steps_per_epoch = total_size // 32 + 1

total_valid_size = 3262
validation_steps = total_valid_size // 32 + 1


def solution_model():
    # model = # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 2 NEURONS ACTIVATED BY SOFTMAX
    #     tf.keras.layers.Dense(2, activation='softmax')
    # ])
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.00005), loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint_path = 'cats_dogs_checkpoint_0624.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 verbose=1,
                                 )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     mode='min',
                                                     patience=5,
                                                     factor=0.8)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10,
                                                     verbose=1)

    model.fit(train_dataset,
              steps_per_epoch=steps_per_epoch,
              epochs=10,
              validation_data=(valid_dataset),
              validation_steps=validation_steps,
              callbacks=[checkpoint, reduce_lr, earlystopping],
              verbose=1
              )

    model.load_weights(checkpoint_path)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.

if __name__ == '__main__':
    model = solution_model()
    model.save("model/cats_dogs_model_0524.h5")
