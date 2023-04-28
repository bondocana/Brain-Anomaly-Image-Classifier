import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import csv
from PIL import Image
from sklearn.naive_bayes import MultinomialNB

# vector cu path-ul fiecarei imagini
images = []

# path-ul directorului data care contine toate imaginile
dir_path = "/kaggle/input/unibuc-brain-ad/data/data"
# lista cu numele tuturor pozelor de forma nume.png
imagePaths = sorted(os.listdir(dir_path))

for imagePath in imagePaths:
    # citesc imaginea din folder-ul data
    img = Image.open("/kaggle/input/unibuc-brain-ad/data/data/" + imagePath).convert('L')
    img_arr = np.array(img)
    # imaginile citite sunt de dimensiune 3
    # le convertesc in dimensiune 1 
    img_arr_flat = np.ravel(img_arr)
    # adaug vectorul de pixeli a imagini la vectorul care le contine pe toate
    images.append(img_arr_flat)

images = np.array(images)

# np.set_printoptions(threshold=sys.maxsize)
# print(images[0])

# impart setul de date
# primele 15000 - date de antrenare
training = images[:15000]
# urmatoarele 20000 - date de validare
validation = images[15000:17000]
# ultimele 5149 - date de test
test = images[17000:22150]

# print(training.shape[0])
# print(validation.shape[0])
# print(test.shape[0])

# citesc label-urile datelor de antrenare
train_labels = []
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as file:
    first_line = file.readline()
    # print(first_line)
    for line in file:
        # citesc pe linii
        line = line.strip()
        # ultimul caracter de pe linie este label-ul
        train_labels.append(int(line[-1]))

train_labels = np.array(train_labels)
# np.set_printoptions(threshold=sys.maxsize)
# print(train_labels.shape[0])

# citesc label-urile datelor de validare
# citirea este asemanatoare cu cea de la cele de antrenare
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

# functie care transforma o matrice care contine pixeli intr-o matrice de aceleasi dim cu valorile 0, 1, .., n-1 reprezentand cele n categori
def valori_bin(array, bins):
    # initializez cu 0 un numpy array auxiliar
    aux = np.zeros(array.shape)
    # iterez prin array si stocheaza index-ul binului potrivit
    for i, elem in enumerate(array):
        aux[i] = np.digitize(elem, bins)
    return aux - 1

# Linspace imparte intervalul [start, stop] in n intervale egale
bins = np.linspace(start = 0, stop = 224, num = 4)
print(bins)

x_train = valori_bin(training, bins)
x_validation = valori_bin(validation, bins)

# modelul Multinomial Naive Bayes
model = MultinomialNB()

# antrenez modelul
model.fit(x_train, train_labels)

# scorul obtinut pe datele de validare
scor = model.score(x_validation, validation_labels)
print(scor)

# prezic label-urile pentru datele de testare
x_test = valori_bin(test, bins)
test_labels = model.predict(x_test)

# np.set_printoptions(threshold=sys.maxsize)
# print(test_labels)

# crearea fisierului csv
i = 0
with open('output.csv', mode='w') as file:
    # linia de inceput "id,class"
    file.write("id,class")
    file.write('\n')
    # pt fiecare imagine de test printez numele si eticheta prezisa de model
    for j in range(17001, 22150):
        n = str(test_labels[i])
        file.write(''.join('0' + str(j) + "," + n))
        file.write('\n')
        i += 1
