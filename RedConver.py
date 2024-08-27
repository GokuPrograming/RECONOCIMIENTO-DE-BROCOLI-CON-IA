import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import pickle
import numpy as np
import os
import cv2
from GenerarDatos import IMAGE_ZISE, CATEGORIAS  # Asumo que tienes un archivo GenerarDatos.py

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

neuronas = [32, 64, 128]
densas = [0, 1, 2]
convpoo = [1, 2, 3]
drop = [0]

# Cargar los datos desde los archivos pickle
x = pickle.load(open("x.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

x = x / 255.0  # Normalizar los datos
y = np.array(y)


def prepare(dir):
    img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_ZISE, IMAGE_ZISE))
    return img.reshape(-1, IMAGE_ZISE, IMAGE_ZISE, 1)


def predecir():
    pred = tf.keras.models.load_model("models/RedConv-n128-cl2-d2-dropout0.keras")  # Cambiado a .keras
    prediction = pred.predict([prepare('rex.jpg')])
    predicted_class = int(prediction[0][0])
    print(CATEGORIAS[predicted_class])


def entrenar():
    if not os.path.exists("models"):
        os.makedirs("models")

    for neurona in neuronas:
        for conv in convpoo:
            for densa in densas:
                for d in drop:
                    NAME = "RedConv-n{}-cl{}-d{}-dropout{}".format(neurona, conv, densa, d)
                    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

                    model = Sequential()
                    model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    if d == 1:
                        model.add(Dropout(0.2))

                    for i in range(conv):
                        model.add(Conv2D(64, (3, 3)))
                        model.add(Activation("relu"))
                        model.add(MaxPooling2D(pool_size=(2, 2)))

                    model.add(Flatten())

                    for i in range(densa):
                        model.add(Dense(neurona))
                        model.add(Activation("relu"))

                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))

                    model.compile(loss="binary_crossentropy",
                                  optimizer="adam",
                                  metrics=['accuracy'])

                    model.fit(x, y, batch_size=30, epochs=10, validation_split=0.3, callbacks=[tensorboard])
                    model.save("models/{}.keras".format(NAME))  # Cambiado a .keras


# Comentar la función de predecir para entrenar primero
# predecir()
#tensorboard --logdir="C:/Users/MEGADETH/Desktop/BROCOLIPROGRAM/logs/"
# Llamar a la función para entrenar
entrenar()
