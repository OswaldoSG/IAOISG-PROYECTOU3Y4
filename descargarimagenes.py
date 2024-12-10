import os
import requests

def descargar_imagenes_iiif(manifest_url, directorio_destino):
    """
    Descarga las imágenes de un manifiesto IIIF.
    """
    # Crear directorio si no existe
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    # Obtener el manifiesto IIIF
    print(f"Obteniendo el manifesto desde: {manifest_url}")
    respuesta = requests.get(manifest_url)
    if respuesta.status_code != 200:
        print(f"Error al obtener el manifiesto: {respuesta.status_code}")
        return

    manifest_data = respuesta.json()

    # Navegar por las estructuras del manifiesto para obtener las imágenes
    imagenes = []
    for canvas in manifest_data.get("sequences", [])[0].get("canvases", []):
        for image in canvas.get("images", []):
            # Extraer URL de la imagen
            image_url = image.get("resource", {}).get("@id", "")
            if image_url:
                imagenes.append(image_url)

    # Descargar las imágenes
    for i, url in enumerate(imagenes):
        try:
            print(f"Descargando imagen: {url}")
            respuesta_img = requests.get(url, stream=True)
            if respuesta_img.status_code == 200:
                # Guardar la imagen con un nombre único
                nombre_archivo = os.path.join(directorio_destino, f"imagen_{i + 1}.jpg")
                with open(nombre_archivo, 'wb') as archivo:
                    for chunk in respuesta_img.iter_content(1024):
                        archivo.write(chunk)
            else:
                print(f"Error al descargar la imagen: {respuesta_img.status_code}")
        except Exception as e:
            print(f"Error con la URL {url}: {e}")

# URL del manifiesto IIIF
manifest_url = "https://www.manuscripta.se/iiif/101024/manifest.json"

# Directorio donde se guardarán las imágenes
directorio_destino = "H:/programas python/antiguo"

# Llamar a la función
descargar_imagenes_iiif(manifest_url, directorio_destino)



