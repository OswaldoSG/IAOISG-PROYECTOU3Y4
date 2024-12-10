from tensorflow.keras.models import load_model
import json

# Carga el modelo desde el archivo .h5
model = load_model('modelo_manuscritos.h5')

# Exporta la arquitectura del modelo a JSON
model_json = model.to_json()

# Guarda la arquitectura en un archivo JSON
with open('modelo.json', 'w') as json_file:
    json_file.write(model_json)

print("La arquitectura del modelo se guardó en 'modelo.json'.")
