import pickle
import numpy as np
from flask import Flask, jsonify, request
from replit.object_storage import Client

app = Flask(__name__)

# Inicjalizacja modelu KNN na poziomie globalnym
knn = None

# Funkcja inicjalizująca model KNN przy starcie serwera
def load_knn_model():
    global knn
    client = Client()
    file_data = client.download_as_bytes("knn_model.pkl")
    if file_data:
        knn = pickle.loads(file_data)
        return knn
    else:
        raise FileNotFoundError("Model nie został znaleziony w object storage.")

# Ładowanie modelu
try:
    knn = load_knn_model()
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return "Serwer Flask działa poprawnie. Model KNN jest gotowy do użycia."

# Endpoint do walidacji obrazka
@app.route('/image', methods=['POST'])
def load_image():
    data = request.get_json()

    # Sprawdzenie, czy dane zostały przesłane
    if data is None or 'image' not in data:
        return jsonify({"error": "Expected 'image' key with vector data."}), 400

    # Walidacja długości wektora
    v_len = len(data['image'])
    if v_len != 784:
        return jsonify({"error": f"Invalid image length. Expected 784 elements, got {v_len}"}), 400

    return jsonify({"message": "Image received and validated"}), 200

# Endpoint do klasyfikacji obrazka
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "Brak obrazka w żądaniu"}), 400

    # Check if the model is loaded
    if knn is None:
        return jsonify({"error": "Model is not loaded."}), 500

    # Pobranie i przetworzenie wektora obrazka
    img_vector = np.array(data['image']).reshape(1, 28 * 28).astype('float32') / 255

    # Predykcja etykiety za pomocą załadowanego modelu
    predicted_label = knn.predict(img_vector)[0]
    return jsonify({"prediction": int(predicted_label)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
