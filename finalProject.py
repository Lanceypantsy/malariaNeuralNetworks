## Author: Lance Barto
## Final Project
## Machine Learning
## Professor: Dr. Jiang Feng
## April 30, 2019

# Imports #####################################################################
import cv2
import os
import numpy as np
import pandas as pd
import sklearn.preprocessing as preproc
from sklearn.metrics import classification_report
from keras.models import Sequential
import tensorflow as tf
import keras
from keras import optimizers
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt

# Read in the images ##########################################################

dataset_path = "cell_images"

# Explore the file to see which directories it contains
class_folders = os.listdir(dataset_path)

data1 = []

# For all of the non-system folders, we build a list of the paths 
# that contain images
for image_class in class_folders:
    # Exclude system files which start with '.'
    if not image_class.startswith('.'):
        image_list = os.listdir('{}/{}'.format(dataset_path, 
                                        image_class))
        # Iterate through the images in each folder, read in the images
        for image_name in image_list:
            if not image_name.startswith('.'):
            
                # For each image in our folder, we first read the image,
                image = cv2.imread('{}/{}/{}'.format(dataset_path, 
                                          image_class, image_name))
                # Next, the image is formatted to 128 x 128
                image = cv2.resize(image, (128, 128), 
                               interpolation=cv2.INTER_NEAREST)
            
                # Next, format the image to Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
                # Append the image to our list of data
                data1.append(image)

data2 = []

# For all of the non-system folders, we build a list of the paths 
# that contain images
for image_class in class_folders:
    # Exclude system folders
    if not image_class.startswith('.'):
        image_list = os.listdir('{}/{}'.format(dataset_path, 
                                        image_class))
        
        # Iterate through each image in th folder
        for image_name in image_list:
            if not image_name.startswith('.'):
            
                # Read in the image
                image = cv2.imread('{}/{}/{}'.format(dataset_path, 
                                          image_class, image_name))
                # Next, the image is formatted to 64 x 64
                image = cv2.resize(image, (64, 64), 
                               interpolation=cv2.INTER_NEAREST)
            
                # Append the image to our list of data
                data2.append(image)

# Exploratory Viualizations ###################################################

# Edge detected positive samples
# Create the figure
plt.figure(num=None, figsize=(8, 6), 
           dpi=80, facecolor='w', edgecolor='k')

# Plot the first ten images
for i in np.arange(12):
    
    # Choose a sample
    sample=prewitt(data1[i])
    
    # Reshape the sample
    sample=sample.reshape((128,128))
    
    # Define the subplot, title, and clear the ticks
    plt.subplot(3, 4, i+1)
    plt.title('Index Number: {}'.format(i))
    plt.imshow(sample,cmap='gray')
    plt.xticks([])
    plt.yticks([])

# Full color positive samples
# Adjust the spacing inbetween the plots and show the subplots
plt.subplots_adjust(left=1, right=2, bottom=1, top=2)
plt.show()

# Create the figure
plt.figure(num=None, figsize=(8, 6), 
           dpi=80, facecolor='w', edgecolor='k')

# Plot the first ten images
for i in np.arange(12):
    
    # Choose a sample
    sample=data2[i]
    
#     # Reshape the sample
#     sample=sample.reshape((28,28))
    
    # Define the subplot, title, and clear the ticks
    plt.subplot(3, 4, i+1)
    plt.title('Index Number: {}'.format(i))
    plt.imshow(sample)
    plt.xticks([])
    plt.yticks([])

# Adjust the spacing inbetween the plots and show the subplots
plt.subplots_adjust(left=1, right=2, bottom=1, top=2)
plt.show()

# Edge detected negative samples
# Create the figure
plt.figure(num=None, figsize=(8, 6), 
           dpi=80, facecolor='w', edgecolor='k')

# Plot the first ten images
for i in np.arange(12):
    
    # Choose a sample
    sample=prewitt(data1[20000 + i])
    
    # Reshape the sample
    sample=sample.reshape((128,128))
    
    # Define the subplot, title, and clear the ticks
    plt.subplot(3, 4, i+1)
    plt.title('Index Number: {}'.format(i))
    plt.imshow(sample,cmap='gray')
    plt.xticks([])
    plt.yticks([])

# Full color negative samples
# Adjust the spacing inbetween the plots and show the subplots
plt.subplots_adjust(left=1, right=2, bottom=1, top=2)
plt.show()

# Create the figure
plt.figure(num=None, figsize=(8, 6), 
           dpi=80, facecolor='w', edgecolor='k')

# Plot the first ten images
for i in np.arange(12):
    
    # Choose a sample
    sample=data2[20000 + i]
    
#     # Reshape the sample
#     sample=sample.reshape((28,28))
    
    # Define the subplot, title, and clear the ticks
    plt.subplot(3, 4, i+1)
    plt.title('Index Number: {}'.format(i))
    plt.imshow(sample,cmap='gray')
    plt.xticks([])
    plt.yticks([])

# Adjust the spacing inbetween the plots and show the subplots
plt.subplots_adjust(left=1, right=2, bottom=1, top=2)
plt.show()

# Create color histograms #####################################################

# Histogram of the edge detected images
hists = []
edges = []

for image in data1:    
    image2 = prewitt(image)
    image3 = preproc.normalize(image2, axis=1, norm='max')
    image3 = image3 * 255
    image4 = image3.astype(int)
    hist = np.histogram(image4, bins = 256)
    edges.append(hist[0][1:255])

# Histogram(s) for the color images 
# Declare empty lists to capture the incoming histograms
this = []
this1 = []
this2 = []

# Iterate through each image and create the histograms
for image in data2: 
    # Capture the histogram for the first color layer
    # ignoring the values at or near zero
    hist = cv2.calcHist([image], [0], None, [255], [1, 255])
    this.append(hist)
    
    # Capture the histogram for the second color layer
    # ignoring the values at or near zero
    hist1 = cv2.calcHist([image], [1], None, [255], [1, 255])
    this1.append(hist1)
    
    # Capture the histogram for the third color layer
    # ignoring the values at or near zero
    hist2 = cv2.calcHist([image], [2], None, [255], [1, 255])
    this2.append(hist2)

# Concatenate all three of the histograms by row
this3 = np.concatenate((this, this1, this2), axis = 1)

# Reshape the array to desired shape. We now have each row representing a sample
# of length 765 - the concatenated histogram length
this3 = this3.reshape(27558, 765)

# Histogram Plots #############################################################

### Positive sample edge-detected histogram
# Define the figure
plt.figure()

# Set the title and axis labels, define the x-axis range
plt.title("Prewitt filtered grayscale histogram, infected")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])

# Plot the histogram distribution
plt.plot(edges[0])

### Negative sample edge-detected histogram
# Define the figure
plt.figure()

# Set the title and axis labels, define the x-axis range
plt.title("Prewitt filtered grayscale histogram, uninfected")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])

# Plot the histogram distribution
plt.plot(edges[20000])

### Positive Sample full-color histogram
# Define the figure
plt.figure()

# Set the title, axis labels, and x-axis range
plt.title("Malaria infected color histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 765])

# Plot the histogram distribution
plt.plot(this3[0])

### Negative Sample full-color histogram
# Define the figure
plt.figure()

# Set the title, axis labels, and x-axis range
plt.title("Uninfected color histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 765])

# Plot the histogram distribution
plt.plot(this3[20000])

# Full-color Histogram Model ##################################################

# Create labels for all models ; 0 - Infected, 1 - Uninfected
labels = np.concatenate((np.repeat(0, 13779), np.repeat(1, 13779)))

# Define the model layers
model = Sequential([
    # Fully connected dense layer with 256 nodes, and relu activation
    Dense(256, input_shape=(765,)),
    Activation('relu'),
    
    # Fully connected dense layer with 128 nodes, and relu activation
    Dense(128, input_shape=(256,)),
    Activation('relu'),
    
    # Single node dense layer with sigmoid activation to create class label
    Dense(1, input_shape=(128,)),
    Activation('sigmoid')
])

# Compile the model using an Adam optimizer, binary cross-entropy loss, 
# and accuracy as the target metric
model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Normalize the data from 0 to 1 (originally 1-255)
this3 = this3 / 255

# Split the data into train/test
(train_X, test_X, train_Y, test_Y) = \
    train_test_split(this3, labels, test_size=.25, random_state = 99)

# Start time
time_in = datetime.now()

# Run the model, append metrics to a variable for plotting
history = model.fit(train_X, train_Y, 
          validation_data=(test_X, test_Y), epochs=20, batch_size=32)

# End time
time_out = datetime.now()

# Training accuracy
plt.plot(history.history['acc'], label='Training Accuracy')

# Testing accuracy
plt.plot(history.history['val_acc'], label='Testing Accuracy')

# Training loss
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Testing Loss')

# Testing loss
plt.legend(bbox_to_anchor=(1, .65), loc='upper left', ncol=1)

# Set the title
plt.title('3-Color Histogram Naive Neural Network')

# Set the y-limit
plt.ylim(0,1)

# Display the plot
plt.show()

# Print execution time
print("Elapsed Time: ", time_out - time_in)

# Edge-detected Histogram Model ###############################################

# Declare the sequential model 
model1 = Sequential([
    
    # Fully connected layer with 254 nodes and sigmoid activation
    Dense(254, input_shape=(254,)),
    Activation('relu'),
    
    # Fully connected layer with 254 nodes and sigmoid activation
    Dense(254, input_shape=(254,)),
    Activation('relu'), 
    
    # Fully connected layer with 128 nodes and sigmoid activation
    Dense(128, input_shape=(254,)),
    Activation('relu'),
    
    # Fully connected single node with sigmoid activation to classify
    Dense(1, input_shape=(128,)),
    Activation('sigmoid')
])

# Compile the odel with an Adam optimizer, binary cross-entropy loss, and 
# accuracy as the target metric
model1.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Parse to array and normalize the data between 0 and 1
edges = np.array(edges)
edges = edges / 254

# Split the data
(train_X, test_X, train_Y, test_Y) = \
    train_test_split(edges, labels, test_size=.25, random_state = 99)

# Parse the type to np.array
train_X = np.array(train_X)
test_X = np.array(test_X)

# Reshape the data for the model 
train_X = train_X.reshape(len(train_X), 254)
test_X = test_X.reshape(len(test_X), 254)

# Start time
time_in = datetime.now()

# Run model, capture metrics
history_greyscale = model1.fit(train_X, train_Y, 
          validation_data=(test_X, test_Y), epochs=20, batch_size=32)

# End time
time_out = datetime.now()

# Training accuracy
plt.plot(history_greyscale.history['acc'], label='Training Accuracy')

# Testing accuracy
plt.plot(history_greyscale.history['val_acc'], label='Testing Accuracy')

# Training loss
plt.plot(history_greyscale.history['loss'], label='Training Loss') 

# Testing loss
plt.plot(history_greyscale.history['val_loss'], label='Testing Loss')

# Add legend to outside of plot
plt.legend(bbox_to_anchor=(1, .65), loc='upper left', ncol=1)

# Set the title and y-axis limits
plt.title('Prewitt Edge Detected Histogram Naive Neural Network')
plt.ylim(0,1)

# Display the plot
plt.show()

# Print execution time 
print("Total elapsed time: ", time_out - time_in)

# Model of the Full-color image data ##########################################

# Declare the model 
model2 = Sequential()

# First 2d Conv layer and activation
model2.add(Conv2D(32, kernel_size=3,
                 padding = 'same',
                 input_shape=(64,64,3)))
model2.add(Activation('relu'))

# Pool to reduce dimentsionality
model2.add(MaxPooling2D(pool_size=2))

# Second 2d Conv layer and activation
model2.add(Conv2D(64, kernel_size=3, 
                padding = 'same'))
model2.add(Activation('relu'))

# Pooling Layer to reduce dimensionality
model2.add(MaxPooling2D(pool_size=2))

# Flatten for the fully connected layer
model2.add(Flatten())

# Add a fully connected layer and activation
model2.add(Dense(512))
model2.add(Activation('relu'))

# Add the final fully connected layer and 
# activate with softmax
model2.add(Dense(1))
model2.add(Activation('sigmoid'))

# Normalize the data between 0 and 1
data2 = np.array(data2) / 255

# Split the data into train/test
(train_X, test_X, train_Y, test_mY) = \
    train_test_split(data2, labels, test_size=.25, random_state=99)

# Parse into np.arrays
train_X = np.array(train_X)
test_X = np.array(test_X)

# Reshape to appropriate shape for modeling
train_X = train_X.reshape(len(train_X), 64, 64, 3)
test_X = test_X.reshape(len(test_X), 64, 64, 3)

# Start time
time_in = datetime.now()

# Compile the model with binary cross-entropy loss, Adam optimizers, and accuracy
# as the target metric
model2.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=0.0001), 
              metrics=["accuracy"])

# Run the model and append the metrics to a variable for plotting
history2 = model2.fit(train_X, train_Y, 
          validation_data=(test_X, test_Y), epochs=10, batch_size=32)

# End time
time_out = datetime.now()

# Training Accuracy
plt.plot(history2.history['acc'], label='Training Accuracy')

# Testing Accuracy
plt.plot(history2.history['val_acc'], label='Testing Accuracy')

# Training loss
plt.plot(history2.history['loss'], label='Training Loss') 

# Testing loss
plt.plot(history2.history['val_loss'], label='Testing Loss')

# Add a legend to the outside of the plot
plt.legend(bbox_to_anchor=(1, .65), loc='upper left', ncol=1)

# Set the title and y-axis limits
plt.title('RGB Convolutional Neural Network')
plt.ylim(0,1)

# Display the plot
plt.show()

# Print the execution time 
print("Elapsed Time: ", time_out - time_in)