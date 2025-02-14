import pickle

import numpy as np
from flask import jsonify, request
from replit.object_storage import Client

from utils.data_formatter import format_json_data
from utils.image_processing import validate_image


# Funkcja inicjalizująca model MLP przy starcie serwera
def load_cnn_model():
    global cnn
    client = Client()
    file_data = client.download_as_bytes("cnn_model.pkl")
    if file_data:
        cnn = pickle.loads(file_data)
        return cnn
    else:
        raise FileNotFoundError("Model CNN nie został znaleziony w object storage.")



def classify_cnn():
    data = request.get_json()

    if not validate_image(data): 
        return jsonify({"error": "Błąd przy walidacji obrazka"})

    # Czy model MLP jest załadowany?
    if cnn is None:
        return jsonify({"error": "Model CNN nie został załadowany."}), 500


    # Pobranie i przetworzenie wektora obrazka
    img_vector = np.array(data['image']).reshape(1, 28, 28).astype('float32') / 255

    # Predykcja etykiety za pomocą załadowanego modelu
    predictions = cnn.predict(img_vector)[0]

    return jsonify(format_json_data(predictions, 'CNN'))