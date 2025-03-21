import pickle

import numpy as np
from flask import jsonify, request
from replit.object_storage import Client

from utils.image_processing import validate_image


# Funkcja ładująca model KNN 
def load_knn_model():
    global knn
    client = Client()
    file_data = client.download_as_bytes("knn_model.pkl")
    if file_data:
        knn = pickle.loads(file_data)
        return knn
    else:
        raise FileNotFoundError("Model KNN nie został znaleziony w object storage.")


def classify_knn():
    data = request.get_json()
    data = request.get_json()

    if not validate_image(data): 
        return jsonify({"error": "Błąd przy walidacji obrazka"})

    # Czy model KNN jest załadowany?
    if knn is None:
        return jsonify({"error": "Model is not loaded."}), 500

    # Pobranie i przetworzenie wektora obrazka
    img_vector = np.array(data['image']).reshape(1, 28 * 28).astype('float32') / 255

    # Predykcja etykiety za pomocą załadowanego modelu
    predicted_label = knn.predict(img_vector)[0]
    return jsonify({"prediction KNN": int(predicted_label)})