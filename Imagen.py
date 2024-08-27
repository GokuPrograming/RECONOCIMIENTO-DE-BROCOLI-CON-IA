import tensorflow as tf
import cv2
import numpy as np
from GenerarDatos import IMAGE_ZISE, CATEGORIAS  # Asegúrate de tener este archivo GenerarDatos.py
 

def prepare(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_ZISE, IMAGE_ZISE))
    return img.reshape(-1, IMAGE_ZISE, IMAGE_ZISE, 1)


def predecir(model_path, image_paths):
    pred = tf.keras.models.load_model(model_path)
    predictions = []

    for image_path in image_paths:
        prediction = pred.predict([prepare(image_path)])
        predicted_class = int(prediction[0][0])
        predictions.append(CATEGORIAS[predicted_class])

    return predictions


if __name__ == "__main__":
    model_path = "models/RedConv-n128-cl1-d2-dropout0.keras"  
    image_paths = ["Bueno1.jpg", "Bueno2.png", "Malo3.jpg", "Malo4.jpg",
                   "Malo5.jpg"] 

    resultados = predecir(model_path, image_paths)

    for i, resultado in enumerate(resultados):
        print(f"La predicción para la imagen {i + 1} es:", resultado)
#funciona=1/5 RedConv-n32-cl1-d0-dropout0
#funciona = 0/5   RedConv-n32-cl1-d1-dropout0
#    funciona=3/5 models/RedConv-n32-cl1-d2-dropout0
#funciona=0//5   RedConv-n32-cl2-d0-dropout0
#funciona=0      RedConv-n32-cl2-d1-dropout0\train
#funcioan = 0    RedConv-n32-cl2-d2-dropout0
#funciona =0      RedConv-n32-cl3-d0-dropout0
#funciona=1/5    RedConv-n128-cl1-d0-dropout0
#funciona = 3/5  RedConv-n128-cl1-d1-dropout0
#funciona= 5/5       RedConv-n128-cl1-d2-dropout0
