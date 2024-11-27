import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from helpers import resize_to_fit


LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


data = []
labels = []

#loop over input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    #load and convert to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #resize letter so it fits in a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)
    
    #add third dimension to image
    image = np.expand_dims(image, axis=2)
    
    #get name of the letter based on folder it's in
    label = image_file.split(os.path.sep)[-2]
    
    #add the letter image and it's label to training data
    data.append(image)
    labels.append(label)
    

#scale the raw pixel intensities to range [0, 1], improves training
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

# Add debug prints to see what's in dataset
unique_labels = sorted(list(set(labels)))
n_classes = len(unique_labels)
print("Unique characters in dataset:", unique_labels)
print("Number of classes:", n_classes)

#split training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size = 0.25, random_state = 0)

#convert the labels (letters) to onehot encodings that keras can work with
lb = LabelBinarizer().fit(labels)  # Fit on all labels first
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Add debug prints
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

#save the mapping from labels to onehot encodings
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)
    
#build the neural network
model = Sequential()

#first convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding = "same", input_shape = (20, 20, 1), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))

#second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

#hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation = "relu"))

#output layer with one node for each possible character
model.add(Dense(n_classes, activation="softmax"))

#ask keras to build the tensorflow model
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

#train neural network
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), batch_size = 32, epochs = 10, verbose = 1)

#save trained model
model.save(MODEL_FILENAME)