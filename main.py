from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS
from replit.object_storage import Client

from services.knn_service import classify_knn, load_knn_model
from services.mlp_service import classify_mlp, load_mlp_model


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Inicjalizacja modeli KNN i MLP na poziomie globalnym
knn = None
mlp = None


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


# Endpoint do klasyfikacji obrazka
@app.route('/classify_knn', methods=['POST'])
def classify_knn_client():
    return classify_knn()

# Endpoint do klasyfikacji obrazka
@app.route('/classify_mlp', methods=['POST'])
def classify_mlp_client():
    return classify_mlp()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)