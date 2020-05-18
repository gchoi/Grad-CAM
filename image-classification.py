#%% Import libraries

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


#%% Define arguments

class DefineArguments():
    def __init__(self,
                 dataset='./dataset/',
                 plot='./plot-images/plot.png',
                 model='./models/my_model.model'):
        self.dataset = dataset
        self.plot = plot
        self.model = model
        return

args = DefineArguments()


#%% Define learning parameters

# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-3
EPOCHS = 50
BS = 45


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
    label = imagePath.split(os.path.sep)[1].split('_')[0]
    
    # load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (384, 288))
    
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
    if idx % 24 == 0:
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

# Partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.20,
                                                  stratify=labels,
                                                  random_state=42)

# Initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15,
                              fill_mode="nearest")


#%% Build the model

# Load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights="imagenet",
                  include_top=False,
                  input_tensor=Input(shape=(288, 384, 3)))

# Construct the head of the model that will be placed on top of the the base model
baseModel.layers
baseModel.input
baseModel.output
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(labels.shape[1], activation="softmax")(headModel)

# Place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
model.summary()

# Loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

for layer in baseModel.layers[-3:]:
    layer.trainable = True

for i in range(len(baseModel.layers)):
    print(f'Layer Name: {baseModel.layers[i].name}, Trainable: {baseModel.layers[i].trainable}')


#%% Compile the model

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


#%% Train the model

# Train the head of the network
print("[INFO] training head...")
H = model.fit_generator(trainAug.flow(trainX, trainY, batch_size=BS),
                        steps_per_epoch=len(trainX) // BS,
                        validation_data=(testX, testY),
                        validation_steps=len(testX) // BS,
                        epochs=EPOCHS)


#%% Prediction & Reports

# Make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# For each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=le.classes_))

# Compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))

acc_total = 0
for i in range(len(cm)):
    acc_total += cm[i, i]

acc = acc_total / total

# Show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))


#%% Plot

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure(figsize=(20, 10))
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args.plot)


#%% Save the model

# Serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(args.model, save_format="h5")