import numpy as np


def preprocess_image(image):
    img = np.array(image).reshape(28,28,1).astype('float32') / 255.0
    return img


## Walidacja danych
def validate_image(data):

    # Sprawdzenie, czy dane zostały przesłane
    if data is None or 'image' not in data:
        print("Oczekiwano klucza 'image' w jsonie.")
        return False

    # Walidacja długości wektora
    v_len = len(data['image'])
    if v_len != 784:
        print("""Nieprawidłowa długość wektora z obrazem.""" 
              f"""Oczekiwano 784 elementów, jest {v_len}.""")
        return False

    if 'image' not in data:
        print("Brak obrazka w żądaniu")
        return False

    return True