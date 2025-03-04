from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS

from services.cnn_service import classify_cnn, load_cnn_model
from services.knn_service import classify_knn, load_knn_model
from services.mlp_service import classify_mlp, load_mlp_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Inicjalizacja modeli KNN i MLP na poziomie globalnym
knn = None
mlp = None
cnn = None


# Ładowanie modelu
try:
    knn = load_knn_model()
    mlp = load_mlp_model()
    cnn = load_cnn_model()
except Exception as e:
    print(f"Błąd ładowania modeli: {e}")

@app.route('/')
def index():
    return "Serwer Flask działa poprawnie. Modele KNN, MLP i CNN zostały wczytane."
    

# Endpoint do klasyfikacji obrazka KNN
@app.route('/classify_knn', methods=['POST'])
def classify_knn_client():
    return classify_knn()

# Endpoint do klasyfikacji obrazka MLP
@app.route('/classify_mlp', methods=['POST'])
def classify_mlp_client():
    return classify_mlp()

# Endpoint do klasyfikacji obrazka CNN
@app.route('/classify_cnn', methods=['POST'])
def classify_cnn_client():
    return classify_cnn()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)