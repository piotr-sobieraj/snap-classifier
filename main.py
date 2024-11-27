import pickle

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
from replit.object_storage import Client

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Inicjalizacja modeli KNN i MLP na poziomie globalnym
knn = None
mlp = None

# Funkcja inicjalizująca model KNN przy starcie serwera
def load_knn_model():
    global knn
    client = Client()
    file_data = client.download_as_bytes("knn_model.pkl")
    if file_data:
        knn = pickle.loads(file_data)
        return knn
    else:
        raise FileNotFoundError("Model KNN nie został znaleziony w object storage.")

# Funkcja inicjalizująca model MLP przy starcie serwera
def load_mlp_model():
    global mlp
    client = Client()
    file_data = client.download_as_bytes("mlp_model.pkl")
    if file_data:
        mlp = pickle.loads(file_data)
        return mlp
    else:
        raise FileNotFoundError("Model MLP nie został znaleziony w object storage.")



# Ładowanie modelu
try:
    knn = load_knn_model()

    mlp = load_mlp_model()
    print (jsonify(mlp.summary()))
except Exception as e:
    print(f"Błąd ładowania modeli: {e}")

@app.route('/')
def index():
    return "Serwer Flask działa poprawnie. Modele KNN i MLP zostały wczytane"

## Wywalić endpoint, dodać walidację do preprocess_image
# Endpoint do walidacji obrazka
@app.route('/image', methods=['POST'])
def load_image():
    data = request.get_json()

    # Sprawdzenie, czy dane zostały przesłane
    if data is None or 'image' not in data:
        return jsonify({"error": "Oczekiwano klucza 'image' w jsonie."}), 400

    # Walidacja długości wektora
    v_len = len(data['image'])
    if v_len != 784:
        return jsonify({"error": f"Nieprawidłowa długość wektora z obrazem. Oczekiwano 784 elementów, jest {v_len}."}), 400

    return jsonify({"message": "Obrazek został odebrany i zwalidowany"}), 200

# Użyć
def preprocess_image(image):
    img = np.array(image).reshape(28,28,1).astype('float32') / 255.0
    return img

# Endpoint do klasyfikacji obrazka
@app.route('/classify_knn', methods=['POST'])
def classify_knn_client():
    return classify_knn()

# Endpoint do klasyfikacji obrazka
@app.route('/classify_mlp', methods=['POST'])
def classify_mlp_client():
    return classify_mlp()


def classify_knn():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "Brak obrazka w żądaniu"}), 400

    # Check if the model is loaded
    if knn is None:
        return jsonify({"error": "Model is not loaded."}), 500

    # Pobranie i przetworzenie wektora obrazka
    img_vector = np.array(data['image']).reshape(1, 28 * 28).astype('float32')

    # Predykcja etykiety za pomocą załadowanego modelu
    predicted_label = knn.predict(img_vector)[0]
    return jsonify({"prediction KNN": int(predicted_label)})

def classify_mlp():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "Brak obrazka w żądaniu"}), 400

    # Check if the model is loaded
    if mlp is None:
        return jsonify({"error": "Model MLP nie został załadowany."}), 500

    
    # Pobranie i przetworzenie wektora obrazka
    img_vector = np.array(data['image']).reshape(1, 28 * 28).astype('float32') / 255
    
    # Predykcja etykiety za pomocą załadowanego modelu
    predictions = mlp.predict(img_vector)[0]
    
    return jsonify(format_json_data(predictions))


def format_json_data(predictions):
    # Znajdź klasę z najwyższym prawdopodobieństwem
    predicted_class = int(np.argmax(predictions.tolist()))

    # Prawdopodobieństwa pozostałych etykiet
    rounded_probabilities = [round(float(p), 5) for p in predictions]

    # Przygotuj wynik
    result = {
      "prediction MLP": predicted_class,
      "probabilities": rounded_probabilities
    }

    return result



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)