from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello from Snap Classifier!'

@app.route('/image', methods=['POST'])
def classify_image():
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


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
