import os
import sys
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, AveragePooling2D, Cropping2D, MaxPooling2D
from keras.models import load_model

#
# This class is used to hold for each data sample the image
# and associated angle. In addition it handle image flipping.
#
# Loading of the image data is handle in the get_sample method
# to allow use of this by generators.
#
class DataSample:
    def __init__(self, image_name, angle, flipped=False):
        self.image_name = image_name
        self.angle = angle
        self.flipped = flipped

    def get_sample(self):
        image = cv2.imread(self.image_name)
        if self.flipped:
            return np.fliplr(image), -self.angle
        else:
            return image, self.angle

def parse_driving_log(path='.', test_size=0.2):
    samples = []
    with open(os.path.join(path,'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # For each line we return 6 samples:
            # * left, center and right images
            # * and their flipped images
            # Center image
            center_name = os.path.join(path, 'IMG', line[0].split('/')[-1])
            center_angle = float(line[3])
            # Left image
            left_name = os.path.join(path, 'IMG', line[1].split('/')[-1])
            left_angle = center_angle+0.1
            # Right image
            right_name = os.path.join(path, 'IMG', line[2].split('/')[-1])
            right_angle = center_angle-0.1
            # Add these three images to the set of samples
            samples.append(DataSample(center_name, center_angle))
            samples.append(DataSample(left_name, left_angle))
            samples.append(DataSample(right_name, right_angle))
            # Add the mirrored versions of these three images (flipped left/right)
            samples.append(DataSample(center_name, center_angle, flipped=True))
            samples.append(DataSample(left_name, left_angle, flipped=True))
            samples.append(DataSample(right_name, right_angle, flipped=True))

    train_samples, validation_samples = train_test_split(samples, test_size=test_size)

    return train_samples, validation_samples

def batch_generator(samples, batch_size=32, path='.'):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Retrieve the individual samples using the
                # sample method of the DataSample class
                image, angle = batch_sample.get_sample()
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def make_driving_model(ch, row, col):

    model = Sequential()

    # Normalize the image data, centered around zero with
    # a small standard deviation 
    model.add(Lambda(lambda x: x/255.0 - 0.5,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))

    # Crop the image to ignore parts of the image that are likely
    # to be irrelevant. 66 pixels ignored at the top, 30 pixels
    # at the bottom.
    model.add(Cropping2D(cropping=((66,30), (0,0))))

    # This convolutional model is aiming to bring the
    model.add(Conv2D(8,(3,3),activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16,(3,3),activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(48,(3,3),activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(3,2)))
    model.add(Conv2D(64,(3,3),activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(3,2)))

    # Once the convolutional layer has been brought to be 1D all
    # the nodes are flattened and progressively driven to only
    # one output node.
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(600, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))

    # The last layer is using a tanh activation to constrain its
    # output to the -1/+1 range.
    model.add(Dense(1, activation="tanh"))

    return model

#
# The path to the data-set is provided on the command line.
#
dataset_path = sys.argv[1]

#
# Load the dataset from the driving log and partition it
# in a training and validation sets.
#
train_samples, validation_samples = parse_driving_log(path=dataset_path,
                                                      test_size=0.3)

N_EPOCHS=50
BATCH_SIZE=32
ch, row, col = 3, 160, 320  # Original image format

model = make_driving_model(ch, row, col)
#model = load_model("model.h5")

# Print the summary of the model.
print(model.summary())

#
# Training dataset generator
#
train_generator = batch_generator(train_samples,
                                  batch_size=BATCH_SIZE,
                                  path=dataset_path)

#
# Validation dataset generator
#
validation_generator = batch_generator(validation_samples,
                                       batch_size=BATCH_SIZE,
                                       path=dataset_path)

#
# Compile and fit the model using the generator
#
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples)/BATCH_SIZE,
                    epochs=N_EPOCHS,
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)/BATCH_SIZE)

#
# Save the resulting model for future use
#
model.save('model.h5')
