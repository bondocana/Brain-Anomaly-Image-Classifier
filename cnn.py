import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import csv
from PIL import Image
from sklearn.naive_bayes import MultinomialNB
import imghdr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Lambda, BatchNormalization, Dropout

images = []
dir_path = "/kaggle/input/unibuc-brain-ad/data/data"
imagePaths = sorted(os.listdir(dir_path))

for imagePath in imagePaths:
    img = Image.open("/kaggle/input/unibuc-brain-ad/data/data/" + imagePath)
    img_arr = np.array(img)
    # img_arr = img_arr / 255 
    images.append(img_arr)

images = np.array(images)

# np.set_printoptions(threshold=sys.maxsize)
# print(images[0])

training = images[:15000]
validation = images[15000:17000]
test = images[17000:22150]

# print(training.shape[0])
# print(validation.shape[0])
# print(test.shape[0])

train_labels = []
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as file:
    first_line = file.readline()
    # print(first_line)
    for line in file:
        line = line.strip()
        train_labels.append(int(line[-1]))

train_labels = np.array(train_labels)
# np.set_printoptions(threshold=sys.maxsize)
# print(train_labels.spahe[0])

validation_labels = []
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as file:
    first_line = file.readline()
    # print(first_line)
    for line in file:
        line = line.strip()
        validation_labels.append(int(line[-1]))

validation_labels = np.array(validation_labels)
# np.set_printoptions(threshold=sys.maxsize)
# print(validation_labels)

# retea neuronala convolutionala
model = Sequential([
    # strat convolutional cu 16 filtre, kernel de 3 x 3, glisare de 1 si functia de activare relu
    # am dat input-ul de (224, 224, 3), fiind primul strat
    Conv2D(16, (3,3), 1, activation = 'relu', input_shape = (224, 224, 3)),
    # normalizez datele pentru a antrena modelul mai eficient
    BatchNormalization(),
    Conv2D(16, (3,3), 1, activation = 'relu'),
    BatchNormalization(),
    # reduce dimensiunea inputului
    MaxPooling2D((2, 2)),
    # impart pixelii la 255.0 pentru a obtine valori intre 0 si 1
    Lambda(lambda x : x / 255.0),
    # cresc numarul de filtre la 32
    Conv2D(32, (3,3), 1, activation = 'relu'),
    BatchNormalization(),
    Conv2D(32, (3,3), 1, activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # 64 de filtre
    Conv2D(64, (3,3), 1, activation = 'relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), 1, activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # 128 de filtre
    Conv2D(128, (3,3), 1, activation = 'relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), 1, activation = 'relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # pune output-ul intr-un vector de dimensiune 1
    Flatten(),
    # 256 de neuroni care conecteaza fiecare neuron din stratul curent la toti cei din stratul precedent
    # functia de activare relu
    Dense(256, activation = 'relu'),
    Dense(256, activation = 'relu'),
    Dense(256, activation = 'relu'),
    # un singur neuron cu functia de activare sigmoid
    Dense(1, activation = 'sigmoid'),
])

# compilez modelul 
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# antrenez modelul in 20 de epoci cu batch_size 64
model.fit(training, train_labels, epochs = 20, batch_size = 64)

# evaluez modelul cu datele de validare
score = model.evaluate(validation, validation_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# model.summary()

# fisierul csv
labels = model.predict(test)
test_labels = []

for label in labels:
    if(label > 0.5):
        test_labels.append(1)
    else:
        test_labels.append(0)

test_labels = np.array(test_labels)
# np.set_printoptions(threshold=sys.maxsize)
# print(test_labels)

i = 0
with open('output.csv', mode='w') as file:
    file.write("id,class")
    file.write('\n')
    for j in range(17001,22150):
        n = str(test_labels[i])
        file.write(''.join('0' + str(j) + "," + n))
        file.write('\n')
        i += 1