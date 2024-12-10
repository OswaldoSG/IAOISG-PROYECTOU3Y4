import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
import pandas as pd

# Ruta al archivo CSV con las imágenes y sus etiquetas
csv_path = 'H:/programas python/imagenes_manuscritos.csv'  # Ruta al archivo CSV

# Función para cargar imágenes desde el CSV
def cargar_imagenes_desde_csv(csv_path, tamaño=(64, 64)):
    imagenes = []
    etiquetas = []

    # Leer el CSV
    df = pd.read_csv(csv_path)

    # Recorrer cada fila del CSV
    for _, row in df.iterrows():
        imagen_path = row['imagen_path']
        etiqueta = row['etiqueta']
        
        # Cargar la imagen
        img = cv2.imread(imagen_path)
        
        if img is None:
            print(f"Error al cargar la imagen: {imagen_path}")
            continue  # Saltar si la imagen no se carga correctamente
        
        # Redimensionar y normalizar la imagen
        img_resized = cv2.resize(img, tamaño)
        img_normalizada = img_resized / 255.0
        
        # Agregar la imagen y su etiqueta
        imagenes.append(img_normalizada)
        etiquetas.append(etiqueta)
    
    return np.array(imagenes), np.array(etiquetas)

# Cargar las imágenes y las etiquetas desde el CSV
imagenes, etiquetas = cargar_imagenes_desde_csv(csv_path)

# Convertir las etiquetas a formato one-hot (porque estamos usando clasificación categórica)
etiquetas_onehot = to_categorical(etiquetas, num_classes=2)

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas_onehot, test_size=0.2, random_state=42)

# Definir el modelo de red neuronal convolucional (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 clases (antiguo vs moderno)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
score = model.evaluate(X_test, y_test)
print(f'Pérdida en test: {score[0]}, Precisión en test: {score[1]}')

# Guardar el modelo entrenado (opcional)
model.save('modelo_manuscritos.h5')

# Predicción para una imagen cargada desde código
def predecir_imagen(imagen_path):
    # Cargar la imagen y redimensionarla
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error al cargar la imagen: {imagen_path}")
        return
    
    img_resized = cv2.resize(img, (64, 64))  # Redimensionar a 64x64
    img_normalizada = img_resized / 255.0  # Normalizar la imagen

    # Convertir la imagen a un formato adecuado para la predicción (agregar la dimensión batch)
    img_expanded = np.expand_dims(img_normalizada, axis=0)

    # Realizar la predicción
    pred = model.predict(img_expanded)
    
    # Determinar la clase con mayor probabilidad
    clase_predicha = np.argmax(pred, axis=1)[0]
    
    if clase_predicha == 0:
        print("La imagen es de un manuscrito antiguo.")
    else:
        print("La imagen es de un manuscrito moderno.")
    
# Ejemplo de predicción
imagen_para_predecir = 'H:/programas python/moderno/modern13.jpg'  # Cambia esto a la ruta de la imagen que quieras predecir
predecir_imagen(imagen_para_predecir)

# Predicción desde la cámara web
def predecir_desde_camara():
    # Abrir la cámara web
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar la imagen de la cámara
        cv2.imshow('Captura de cámara', frame)

        # Esperar hasta que se presione una tecla específica para hacer la predicción
        key = cv2.waitKey(1) & 0xFF
        
        # Si el usuario presiona "Enter", realiza la predicción
        if key == 13:  # 13 es el código ASCII de la tecla "Enter"
            # Redimensionar la imagen para la predicción
            img_resized = cv2.resize(frame, (64, 64))
            img_normalizada = img_resized / 255.0  # Normalizar la imagen
            img_expanded = np.expand_dims(img_normalizada, axis=0)

            # Realizar la predicción
            pred = model.predict(img_expanded)
            
            # Mostrar el resultado de la predicción
            clase_predicha = np.argmax(pred, axis=1)[0]
            if clase_predicha == 0:
                cv2.putText(frame, "Manuscrito Antiguo", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Manuscrito Moderno", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostrar la predicción en la cámara
            cv2.imshow('Predicción en tiempo real', frame)
        
        # Salir con la tecla 'q'
        if key == ord('q'):
            break

    # Cerrar la cámara y las ventanas
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para hacer predicciones en tiempo real desde la cámara
predecir_desde_camara()