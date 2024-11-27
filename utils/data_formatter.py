import numpy as np


def format_json_data(predictions):
    # Znajdź klasę z najwyższym prawdopodobieństwem
    predicted_class = int(np.argmax(predictions.tolist()))

    # Prawdopodobieństwa pozostałych etykiet
    rounded_probabilities = [round(float(p), 5) for p in predictions]

    # Słownik prawdopodobieństw
    probabilities_dict = {str(i): rounded_probabilities[i] for i in range(len(rounded_probabilities))}

    # Przygotuj wynik
    result = {
      "prediction MLP": predicted_class,
      "probabilities": probabilities_dict
    }

    return result