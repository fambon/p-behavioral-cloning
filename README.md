# Behavioral Cloning 

## Project Goals

The goals of this project were the following:

* Use the driving simulator to collect data of good driving behavior
* Build, a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the first track without leaving the road


[//]: # (Image References)

[image01]: ./images/left-center-right.png "Driving images"
[image02]: ./images/left.jpg   "Left image"
[image03]: ./images/center.jpg "Center image"
[image04]: ./images/right.jpg  "Right image"

---

### Files included

The project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 showing a recording of a lap around the first of the two circuits
* README.md the present file

### How to verify the model behavior

Using the Udacity provided simulator and the drive.py script, the car
can be driven autonomously around the track by executing:

```sh
python drive.py model.h5
```

## Model architecture and parameters

### Model architecture

The model consists of a convolutional neural network which reduces
progressively the RGB image resolution to a single horizontal 1D stripe.
It is based on the NVIDIA model architecture.

The model uses maximum pooling after each convolutional layer
to progressively reduce the 2D dimensions of the image.

The model uses RELU activation layers to introduce nonlinearities.

For the output layer the choice was made to constrain the output steering
command to a numerical range of -1.0 to +1.0 by using tanh activation.

| Layer (type)                   | Output Shape  | Param #   |
|:------------------------------:|:-------------:|:---------:|
| lambda_1 (Lambda)              | 160x320x3     | 0         |
| cropping2d_1 (Cropping2D)      | 64x320x3      | 0         |
| conv2d_1 (Conv2D)              | 64x320x8      | 608       |
| max_pooling2d_1 (MaxPooling2D) | 32x160x8      | 0         |
| conv2d_2 (Conv2D)              | 32x160x16     | 3216      |
| max_pooling2d_2 (MaxPooling2D) | 16x80x16      | 0         |
| conv2d_3 (Conv2D)              | 16x80x48      | 6960      |
| max_pooling2d_3 (MaxPooling2D) | 5x40x48       | 0         |
| conv2d_4 (Conv2D)              | 5x40x64       | 27712     |
| max_pooling2d_4 (MaxPooling2D) | 1x20x64       | 0         |
| flatten_1 (Flatten)            | 1280          | 0         |
| dense_1 (Dense)                | 600           | 768600    |
| dense_2 (Dense)                | 200           | 120200    |
| dense_3 (Dense)                | 100           | 20100     |
| dense_4 (Dense)                | 50            | 5050      |
| dense_5 (Dense)                | 1             | 51        |


#### Normalization

The images are normalized in the model using a Keras lambda layer. 

#### Cropping

In addition the image is cropped to only retain a part of the image that
is relevant to make driving decisions.

#### Model implementation

Here is for reference the model source code:

```python
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
```

### Model parameters tuning

#### Overfitting handling

The model contains dropout layers in order to reduce overfitting (see code above). 

Many variations of dropout ratio were tried ranging from 50% down to 10% and the
best overfitting control was achieved with a dropout ratio of 20%.

#### Learning rate tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### Number of epochs

After many experimentations I came to the conclusion that reasonably stable
driving was achieved when an MSE of about 0.0008 or less was achieved without
notable overfitting.

The best result was achieved with a number of epochs of 50.

## Training data

### Poor training data quality

During the early phase of this project the driving data initially collected was
of poor quality.

First a mistake was made as the simulator wasn't run in the best resolution
at first and this caused a loss of quality in the recorded images.

In addition the intial data set also had the following issues:

* the driving style was not consistent which meant that under similar conditions
  the driving the driving data samples could contradict each other.
* driving using the keyboard leads to all or nothing steering commands
  in the training data.

The consequence was that the mean square error (MSE) on the training and validation
data set were inherently limited by a floor, and could not possibly become sufficiently
small to translate into stable driving.

It became clear that improving the quality of the driving data had a significant
influence on the outcome.

### Collecting a better data set

Here are some best practices recommended to collect a quality driving data set using
the Udacity simulator:

* turn on the maximum quality settings in order to collect the best quality of images
* use a playstation controller (or similar) to improve the quality of the steering
  commands recorded
* hand the task of test driving to a teenager (they are better at this and you know it...)
* specify to your test driver the driving behavior you expect
* have the test driver practice before recording a good session
* record several test drives, review them and keep the best one

The specification for the test driver (my youngest son) was the following:

* smooth drive, with no sudden course correction
* well centered drive long the full course


###  Data augmentation

The training data was augmented in two ways:

* use all three images produced by the simulator: left, right and center
* flip left and right for each image

For left and right image +/- 0.1 is added to the steering command associated
to the center image.

![Left, center and right image][image01]

This value of 0.1 was determined to be a good value after numerous experiments.

![+0.1 steering correction][image02] ![Center steering][image03] ![-0.1 steering correction][image04]

When flipping images the sign of the associated steering command is changed.


## Training Strategy

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering
angle data into a training and validation set.

At the end of the process, the vehicle is able to drive autonomously around the track
without leaving the road.

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if
the model was over or under fitting.
