import pickle

import numpy as np
from flask import jsonify, request
from replit.object_storage import Client

from utils.data_formatter import format_json_data
from utils.image_processing import validate_image


# Funkcja ładująca model MLP
def load_mlp_model():
    global mlp
    client = Client()
    file_data = client.download_as_bytes("model_mlp.pkl")
    if file_data:
        mlp = pickle.loads(file_data)
        return mlp
    else:
        raise FileNotFoundError("Model MLP nie został znaleziony w object storage.")



def classify_mlp():
    data = request.get_json()

    if not validate_image(data): 
        return jsonify({"error": "Błąd przy walidacji obrazka"})

    # Czy model MLP jest załadowany?
    if mlp is None:
        return jsonify({"error": "Model MLP nie został załadowany."}), 500


    # Pobranie i przetworzenie wektora obrazka
    img_vector = np.array(data['image']).reshape(1, 28 * 28).astype('float32') / 255

    # Predykcja etykiety za pomocą załadowanego modelu
    predictions = mlp.predict(img_vector)[0]

    return jsonify(format_json_data(predictions, 'MLP'))