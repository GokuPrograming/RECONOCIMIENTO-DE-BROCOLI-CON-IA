import os
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import random as rm
import numpy as np
import pickle


def Generar_datos():
    # definir una lista para guardar
    data = []
    for categoria in CATEGORIAS:
        path = os.path.join(DATADIR, categoria)
        valor = CATEGORIAS.index(categoria)
        listdir = os.listdir(path)
        # recorrer cada imagen
        for i in tqdm(range(len(listdir)), desc=categoria):
            image_nombre = listdir[i]
            try:
                imagen_ruta = os.path.join(path, image_nombre)
                imagen = cv2.imread(imagen_ruta, cv2.IMREAD_GRAYSCALE)
                imagen = cv2.resize(imagen, (IMAGE_ZISE, IMAGE_ZISE))
                data.append([imagen, valor])
            except Exception as e:
                pass
    rm.shuffle(data)
    x = []
    y = []

    for i in tqdm(range(len(data)), desc="Procesamiento"):
        par = data[i]
        x.append(par[0])
        y.append(par[1])

    x = np.array(x).reshape(-1, IMAGE_ZISE, IMAGE_ZISE, 1)

    pickle_out = open("x.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()
    print("archivo x crreado")

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    print("archivo y crreado")


CATEGORIAS = ["Bueno", "Malo"]
IMAGE_ZISE = 100

if __name__ == "__main__":
    # definir el directorio
    DATADIR = "C:\\Users\\MEGADETH\\Desktop\\BROCOLIPROGRAM\\Brocoli"

    Generar_datos()


#Objetivo: Preprocesar imágenes para un modelo de aprendizaje automático. Las imágenes se leen, redimensionan, y se almacenan en un formato adecuado para entrenamiento.
#Resultado: Dos archivos pickle (x.pickle y y.pickle) que contienen las imágenes y sus etiquetas correspondientes. Estos archivos se utilizan para entrenar y evaluar modelos de aprendizaje automático.