import os
import csv
from random import shuffle
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
import tensorflow as tf

### Load Image Data
PATH = 'data/'
samples = []
with open(PATH+'driving_log2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)
# print (train_samples[0]) #Verify data

### Batch generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = PATH+'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
			
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
			
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
# next(train_generator)

# ch, row, col = 3, 160, 320  # Trimmed image format
### Get input image shape from training samples
input_shape = cv2.imread(PATH+'IMG/'+train_samples[0][0].split('/')[-1]).shape
# print ('input_shape',input_shape)
def resized(img):
	'''
	This helper function will make drive.py can import tensorflow 
	'''
	import tensorflow as tf
	return tf.image.resize_images(img, (66, 200))
### Build Model	
def LeNet5(input_shape):
	'''
	This model will preprocess the data included crop and normalized and then feed to LeNet-5 model
	'''
	### Build Model
	model = Sequential()
	# Cropping2D layer might be useful for choosing an area of interest that excludes the sky and/or the hood of the car.
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))
	# Resize Data to fit the model
	# model.add(Lambda(lambda x: tf.image.resize_images(x, (32, 32))))
	model.add(Lambda(resized))
	# Preprocess incoming data, centered around zero with small standard deviation 
	model.add(Lambda(lambda x: x/127.5 - 1.))
	### LeNet5 Model
	model.add(Convolution2D(6, 5, 5,border_mode='valid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Convolution2D(16, 5, 5,border_mode='valid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Activation('relu'))
	model.add(Dense(84))
	model.add(Activation('relu'))	
	# Output only 1, bacuase it's a regression insteadd of  classification	
	model.add(Dense(1))	
	
	return model

def CNN(input_shape):
	'''
	This CNN model from Nvidia End to End Learning for Self-Driving Cars
	'''
	### Build Model
	model = Sequential()
	# Cropping2D layer might be useful for choosing an area of interest that excludes the sky and/or the hood of the car.
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))
	# Resize Data to fit the model
	## Kareas can't recoginze tensorflow, need to build a helper to resize
	# model.add(Lambda(lambda x: tf.image.resize_images(x, (66, 200))))
	model.add(Lambda(resized))
	# Preprocess incoming data, centered around zero with small standard deviation 
	model.add(Lambda(lambda x: x/127.5 - 1.))
	### CNN Model
	model.add(Convolution2D(24, 5, 5,border_mode='valid', subsample=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Convolution2D(36, 5, 5,border_mode='valid', subsample=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))	
	model.add(Convolution2D(48, 5, 5,border_mode='valid', subsample=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3,border_mode='valid'))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))	
	model.add(Convolution2D(64, 3, 3,border_mode='valid'))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))	
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dense(50))
	model.add(Activation('relu'))	
	model.add(Dense(10))
	model.add(Activation('relu'))	
	# Output only 1, bacuase it's a regression insteadd of classification	
	model.add(Dense(1))	
	
	return model
	
EPOCHS = 10	
FINE_Tune = 1
### The CNN model working on most of times, just fail on some turn. Using fine-tune method to improve exist model.
if FINE_Tune:
	model = load_model('model.h5')
### Build a new model from scratch
else:	
	model = CNN(input_shape)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
					samples_per_epoch= len(train_samples), 
					validation_data=validation_generator, 
					nb_val_samples=len(validation_samples), 
					nb_epoch=EPOCHS)
model.save('model.h5')		
model.summary()	