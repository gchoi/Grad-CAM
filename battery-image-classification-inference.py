#%% Import libraries

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from imutils import paths
import numpy as np
import cv2
import os
import time


#%% Define arguments

class DefineArguments():
    def __init__(self,
                 dataset='./dataset/batteries',
                 plot='./plot-images/plot.png',
                 model='./models/battery_model.model'):
        self.dataset = dataset
        self.plot = plot
        self.model = model
        return

args = DefineArguments()


#%% Define iniference parameters

# initialize the initial learning rate, number of epochs to train for, and batch size
BS = 8
RESIZE_W = 750
RESIZE_H = 500


#%% List of images

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

imagePaths = list(paths.list_images(args.dataset))
data = []
labels = []

# Loop over the image paths
idx = 0
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[1].split('-')[0]
    
    # load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (RESIZE_W, RESIZE_H))
    
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
    if idx % 10 == 0:
        print(f'Current class being proceesed: {label}')
    
    idx += 1


# Convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)

# Perform one-hot encoding on the labels
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)


#%% Load the model

print("[INFO] loading pre-trained network...")
loaded_model = load_model(args.model)
loaded_model.summary()


#%% Evaluate with train set

# Make predictions on the testing set
print("[INFO] evaluating network...")
start = time.time()
predIdxs = loaded_model.predict(data, batch_size=BS)
end = time.time()
print(f"Average Processing Time: {(end - start) / len(predIdxs)} sec.")
predIdxs = np.argmax(predIdxs, axis=1)

# Show a nicely formatted classification report
print(classification_report(labels.argmax(axis=1), predIdxs, target_names=le.classes_))

# Compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(labels.argmax(axis=1), predIdxs)
total = sum(sum(cm))

acc_total = 0
for i in range(len(cm)):
    acc_total += cm[i, i]

acc = acc_total / total

# Show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))