from flask import Flask
from flask_cors import CORS  # Import CORS

from services.cnn_aug_service import classify_cnn_aug, load_cnn_aug_model
from services.knn_service import classify_knn, load_knn_model
from services.mlp_service import classify_mlp, load_mlp_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Inicjalizacja modeli KNN i MLP na poziomie globalnym
knn = None
mlp = None
# cnn = None
cnn_aug = None


# Ładowanie modelu
try:
    knn = load_knn_model()
    mlp = load_mlp_model()
    cnn_aug = load_cnn_aug_model()
except Exception as e:
    print(f"Błąd ładowania modeli: {e}")

@app.route('/')
def index():
    return "Serwer Flask działa poprawnie. Modele KNN, MLP i CNN i CNN augmented zostały wczytane."
    

# Endpoint do klasyfikacji obrazka KNN
@app.route('/classify_knn', methods=['POST'])
def classify_knn_client():
    return classify_knn()

# Endpoint do klasyfikacji obrazka MLP
@app.route('/classify_mlp', methods=['POST'])
def classify_mlp_client():
    return classify_mlp()

# Endpoint do klasyfikacji obrazka CNN augmented
@app.route('/classify_cnn_aug', methods=['POST'])
def classify_cnn_aug_client():
    return classify_cnn_aug()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)