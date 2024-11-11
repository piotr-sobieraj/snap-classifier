import pickle
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, jsonify, request
from replit.object_storage import Client


app = Flask(__name__)


@app.route('/')
def index():
    client = Client()

    # Pobranie modelu KNN jako dane binarne
    try:
        file_data = client.download_as_bytes("knn_model.pkl")
    except Exception as e:
        return f"Nie udało się pobrać modelu: {e}", 500

    # Załaduj model KNN za pomocą pickle
    if file_data:
        knn_loaded = pickle.loads(file_data)
        return "Model został załadowany i jest gotowy do użycia."
    else:
        return "Model nie został znaleziony w object storage.", 404


@app.route('/image', methods=['POST'])
def load_image():
    data = request.get_json()

    # Sprawdzenie, czy dane zostały przesłane
    if data is None:
        return jsonify({"error": "No data provided"}), 400

    # Sprawdzenie, czy przesłany wektor jest odpowiedniej długości
    if 'image' not in data:
        return jsonify({"error": 'Expected an image in format of a vector.'}), 400
        
    v_len = len(data['image'])
    if v_len != 784:
        return jsonify({"error": f"Invalid image length. Expected 784 elements for MNIST image, got {v_len}"}), 400

    # Tutaj można wstawić kod klasyfikacyjny, np. wywołanie modelu predykcyjnego
    # W tym przykładzie zwracamy tylko informację zwrotną z potwierdzeniem
    # Przykład: predykcja = model.predict(data['vector'])

    # Zwracamy informację zwrotną (tutaj tylko przykładowa odpowiedź)
    response = {"message": "Image received", "image": sum(data['image'])}
    return jsonify(response), 200

# Endpoint do klasyfikacji obrazka
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "Brak obrazka w żądaniu"}), 400

    # Pobranie i przetworzenie wektora obrazka
    img_vector = np.array(data['vector']).reshape(1, 28 * 28).astype('float32') / 255

    # Predykcja etykiety za pomocą załadowanego modelu
    predicted_label = knn.predict(img_vector)[0]
    return jsonify({"prediction": int(predicted_label)})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
