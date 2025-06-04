import pickle

import numpy as np
from flask import jsonify, request
from replit.object_storage import Client

from utils.data_formatter import format_json_data
from utils.image_processing import validate_image


# Funkcja ładująca model CNN
def load_cnn_aug_model():
    global cnn_aug
    client = Client()
    file_data = client.download_as_bytes("model_cnn_aug.pkl")
    if file_data:
        cnn_aug = pickle.loads(file_data)
        return cnn_aug
    else:
        raise FileNotFoundError("Model CNN augmented nie został znaleziony w object storage.")



def classify_cnn_aug():
    data = request.get_json()

    if not validate_image(data): 
        return jsonify({"error": "Błąd przy walidacji obrazka"})

    # Czy model CNN jest załadowany?
    if cnn_aug is None:
        return jsonify({"error": "Model CNN augmented nie został załadowany."}), 500

    # Pobranie i przetworzenie wektora obrazka
    img_vector = np.array(data['image']).reshape(1, 28, 28).astype('float32') / 255
    
    # Predykcja etykiety za pomocą załadowanego modelu
    predictions = cnn_aug.predict(img_vector)[0]

    return jsonify(format_json_data(predictions, 'CNN_AUG'))