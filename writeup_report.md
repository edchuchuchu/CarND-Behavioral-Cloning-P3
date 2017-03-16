# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Full-tune the model with additional  recovery data for fail turn.
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/Nvidia-CNN-model.jpg "Model Image"
[image2]: ./report_images/center_2017_02_19_00_20_43_252.jpg "Center Lane Image1"
[image3]: ./report_images/center_2017_02_19_00_20_58_120.jpg "Center Lane Image2"
[image4]: ./report_images/center_2017_02_19_00_20_50_972.jpg "Recovery Image1"
[image5]: ./report_images/center_2017_02_19_00_20_53_231.jpg "Recovery Image2"
[image6]: ./report_images/center_2017_02_13_22_20_13_654.jpg "Fail location Image"
[image7]: ./report_images/center_2017_02_19_00_15_59_850.jpg "Recovery Case1 Image1"
[image8]: ./report_images/center_2017_02_19_00_15_59_924.jpg "Recovery Case1 Image2"
[image9]: ./report_images/center_2017_02_19_00_15_59_998.jpg "Recovery Case1 Image3"
[image10]: ./report_images/center_2017_02_19_00_17_25_509.jpg "Recovery Case2 Image1"
[image11]: ./report_images/center_2017_02_19_00_17_25_582.jpg "Recovery Case2 Image2"
[image12]: ./report_images/center_2017_02_19_00_17_25_654.jpg "Recovery Case2 Image3"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model.h5 containing a trained convolution neural network 
* run1.mp4 to show my model successfully drives around track one without leaving the road
* drive.py for driving the car in autonomous mode
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, execute the following command to drive the car in track one autonomously
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the Nvidia convolution neural network. The file shows the pipeline I used generator for training and validating to bulid and fine-tune the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

At the beginning, I tried to use Keras to build the LeNet-5 model. However, it can’t even pass the first turn. I realized that LeNet-5 model maybe work on classification case instead of regression. 
```python
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
```
Then, I use the Nvidia “End to End Learning for Self-Driving Cars” CNN structure to build the model with RELU and drop method. And preprocess the data with normalized method to equal variance and zero mean, and also crop the unwanted area to help the model focus on useful information.
```python
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
```
#### 2. Attempts to reduce overfitting in the model

Dropout method is using in this model to prevent overfitting. 
```python
model.add(Convolution2D(64, 3, 3,border_mode='valid'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
```
and split 20% training dataset to validation data in order to verify overfitting.
```python
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)
```
#### 3. Model parameter tuning

The model is using an adam optimized.
And I used the default learning rate.

```python
model.compile(loss='mse', optimizer='adam')
```

#### 4. Appropriate training data

Using Udacity provided training data at the beginning to build the initial model.	
Second, Driving the car in the center lane for both clockwise and count-clockwise direction each 2 laps.	
Third, added 2 recovery laps for both clockwise and count-clockwise each.	
In the end, added more recovery data for the fail location in autonomous mode.	
For more details, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Step1 Model choosing,	
    I used LeNet at the beginning since it works pretty well on recognize image.
The self-driving is also using image as training data.
I think it should be work wells. However, it can’t even pass the first turn.
It makes me realize that maybe LeNet works well on pattern match doesn’t mean it can also work well on autonomous. 
So I drop the LeNet structure and use Nvidia CNN structure to re-build my model.

Step2 Training data,	
    It works well at the first turn but fail on the second turn.
The reason may be that training data is not enough to train it.
In order to get more data to train the model, I drove the car in training model and keep the car at the center location for both clockwise and counter-clockwise direction and 2 laps each.
Using this data and pre-provided data to re-train the model.
It works very well! Only fail on one turn, and I found out the fail reason the model doesn’t  know how to recover from side to the center.
So I drive the car again to get more training data. This time I created lots of recovery case to help the model learn how to recovery from side to the center. 
After that, I also take a look the csv file to make sure there is no unwanted data like driving from center to the side will feed to the model. 
Because we don't want the model learn those behaviors.
And I use all the data I have right now to train the model again. 
But it came out worse, the model fail on more turn than the previous model.

Setup3 Fine-tune,	
    It’s obvious the previous model is better. 
I'm thinking about using the fine-tune method which I learned from transfer-learning to improve the previous model.
So I use the previous model which only fails on one turn as my pre-trained model.
And feed the recovery data as new training dataset to fine-tune the model.
It made the new model work on track 1 which means it will not drive out of road in autonomous mode.
In order to make sure the model really works on track one, I try several different cases to make sure the car can success keep on the road.
However, it will fail on one specify location if I increase the speed.
So I add more center lane, recovery lap, clockwise and counter-clockwise data at this location to fine-tune the model.
It can help to fix the issue, and work on any speed at track one in autonomous mode. 


#### 2. Final Model Architecture

I use Nvidia CNN structure model at my final approach. (see the following fig)

![alt text][image1]
```python
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
```

#### 3. Creation of the Training Set & Training Process


Using pre-provided data first.
And then added 2 lap on track which driving at the center of lane.   
![alt text][image2]
![alt text][image3]
   
I found out the car can't recover from side to the center.
To solve this issue, I add recovery lap data.   
![alt text][image4]
![alt text][image5]
   
Then the car will on fail on specify location.   
![alt text][image6]
   
I added more recovery case on this location to train this model.
   
From right to the center:   
![alt text][image7]   
![alt text][image8]   
![alt text][image9]   
   
From left to the center:   
![alt text][image10]   
![alt text][image11]   
![alt text][image12]   
