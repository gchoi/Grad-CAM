"""
@Description: Grad-CAM for class-discriminative localization map
@Date: 5 May., 2020
@Author: Alex Choi
@Contact: cinema4dr12@gmail.com
"""

#%% Import libraries
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from imutils import paths
import numpy as np
import cv2
import os
import random

from gradcam import GradCAM
import imutils


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


#%% Predict for randomly selected image
idx = random.randint(0, len(imagePaths))
img_path = imagePaths[idx]
label = img_path.split(os.path.sep)[1].split('-')[0]

orig_image = cv2.imread(img_path)
orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(orig_image, (RESIZE_W, RESIZE_H))
image = np.expand_dims(image, axis=0)
image = np.array(image) / 255.0

predIdxs = loaded_model.predict(image)
predIdxs = np.argmax(predIdxs, axis=1)

print(f"Input Class: {label}, Predicted Clss: {le.classes_[predIdxs][0]}")

cam = GradCAM(loaded_model, predIdxs[0])
heatmap = cam.compute_heatmap(image)

heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig_image, alpha=0.5)

# Draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2)

# Display the original image and resulting heatmap and output image to our screen
output = np.vstack([orig_image, heatmap, output])
output = imutils.resize(output, height=700)

cv2.imwrite(f'./outputs/{label}.png', output)