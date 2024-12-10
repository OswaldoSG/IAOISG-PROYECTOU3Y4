import os
import numpy as np
import cv2
import pandas as pd

# Directorios de las imágenes
antiguo_dir = 'H:/programas python/antiguo'  # Cambia esto a la ruta de tus imágenes de manuscritos antiguos
moderno_dir = 'H:/programas python/moderno'  # Cambia esto a la ruta de tus imágenes de manuscritos modernos

# Función para cargar y redimensionar imágenes
def cargar_imagenes_a_csv(directorio, etiqueta, archivo_csv):
    # Recorrer todos los archivos en el directorio
    for archivo in os.listdir(directorio):
        if archivo.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')):  # Filtrar por extensión de imagen
            # Ruta completa del archivo
            imagen_path = os.path.join(directorio, archivo)
            print(f"Cargando imagen: {imagen_path}")  # Imprimir la ruta para depuración

            # Cargar la imagen
            img = cv2.imread(imagen_path)
            
            # Verificar si la imagen se carga correctamente
            if img is None:
                print(f"Error al cargar la imagen: {imagen_path}")
                continue  # Saltar esta imagen si no se pudo cargar
            
            # Redimensionar la imagen a 64x64 píxeles
            img_resized = cv2.resize(img, (64, 64))
            
            # Aplanar la imagen para convertirla en un vector (64*64*3 = 12288 elementos)
            img_vector = img_resized.flatten()

            # Agregar la imagen a los datos, incluyendo la ruta de la imagen
            archivo_csv.write(','.join(map(str, img_vector)) + ',' + str(etiqueta) + ',' + imagen_path + '\n')  # Escribir la imagen, su etiqueta y la ruta

# Abrir archivo CSV para escribir
with open('imagenes_manuscritos.csv', 'w') as archivo_csv:
    # Escribir la cabecera (títulos de las columnas)
    cabecera = [f'pixel_{i}' for i in range(12288)] + ['etiqueta', 'imagen_path']  # Agregar 'imagen_path' a la cabecera
    archivo_csv.write(','.join(cabecera) + '\n')

    # Cargar imágenes de manuscritos antiguos (etiqueta 0)
    cargar_imagenes_a_csv(antiguo_dir, 0, archivo_csv)

    # Cargar imágenes de manuscritos modernos (etiqueta 1)
    cargar_imagenes_a_csv(moderno_dir, 1, archivo_csv)

print("Imágenes convertidas y guardadas en CSV con éxito.")
