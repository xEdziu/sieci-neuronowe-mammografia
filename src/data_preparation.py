from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Na początku pliku data_preparation.py
encoder = None
diagnostic_feature_names = []

def process_diagnostic_features(all_data, diagnostic_features):
    """
    Przetwarza dane diagnostyczne: koduje kategoryczne i normalizuje liczbowe.
    """
    print('Przetwarzam dane diagnostyczne -', os.path.basename(__file__))
    diagnostics = all_data[diagnostic_features].fillna("UNKNOWN")  # Uzupełnij braki

    # Weryfikacja kolumn
    categorical_columns = ['mass shape', 'mass margins', 'calc type', 'calc distribution']
    numerical_columns = ['breast density']

    # Zamiana 'UNKNOWN' na wartość domyślną
    diagnostics['breast density'] = diagnostics['breast density'].replace('UNKNOWN', np.nan)
    median_value = diagnostics['breast density'].median()
    diagnostics['breast density'] = diagnostics['breast density'].fillna(median_value)

    missing_categorical = [col for col in categorical_columns if col not in diagnostics.columns]
    missing_numerical = [col for col in numerical_columns if col not in diagnostics.columns]

    if missing_categorical:
        print(f"Brakujące kolumny kategoryczne: {missing_categorical}")
    if missing_numerical:
        print(f"Brakujące kolumny liczbowe: {missing_numerical}")

    # One-Hot Encoding dla kolumn tekstowych
    print('One-Hot Encoding dla kolumn tekstowych -', os.path.basename(__file__))
    encoder = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(diagnostics[categorical_columns]).toarray()

    # Normalizacja kolumn liczbowych
    print('Normalizacja kolumn liczbowych -', os.path.basename(__file__))
    scaler = MinMaxScaler()
    scaled_numerical = scaler.fit_transform(diagnostics[numerical_columns])

    # Dodanie cechy dla 'UNKNOWN'
    for col in categorical_columns:
        diagnostics[f'{col}_is_unknown'] = (diagnostics[col] == 'UNKNOWN').astype(float)
    unknown_features = diagnostics[[f'{col}_is_unknown' for col in categorical_columns]].to_numpy()

    # Upewnienie się, że unknown_features ma 2 wymiary
    unknown_features = unknown_features.reshape(-1, unknown_features.shape[1])

    # Łączenie zakodowanych i znormalizowanych danych
    print('Łączenie zakodowanych i znormalizowanych danych -', os.path.basename(__file__))
    X_diagnostic = np.hstack([scaled_numerical, encoded_categorical, unknown_features])

    # Debugowanie
    print(f"Rozmiar danych diagnostycznych po przetworzeniu: {X_diagnostic.shape}")
    print(f"Pierwsze 5 wierszy danych diagnostycznych:\n{X_diagnostic[:5]}")

    return X_diagnostic


def load_and_prepare_data(img_dir, csv_dir, img_size=(512, 512), test_size=0.2):
    """
    Wczytuje i łączy dane z wielu plików CSV, przetwarza obrazy i generuje etykiety.

    Args:
        img_dir (str): Ścieżka do katalogu z obrazami JPEG.
        csv_dir (str): Ścieżka do katalogu z plikami CSV.
        img_size (tuple): Docelowy rozmiar obrazów (szerokość, wysokość).

    Returns:
        tuple: Obrazy (X_image), dane diagnostyczne (X_diagnostic), etykiety (y).
    """
    # Lista plików CSV do wczytania
    csv_files = [
        "calc_case_description_train_set.csv",
        "mass_case_description_train_set.csv"
    ]
    
    print('Wczytuję dane z plików CSV -', os.path.basename(__file__))
    # Wczytanie i połączenie danych z plików CSV
    all_data = pd.concat(
        [pd.read_csv(os.path.join(csv_dir, csv_file)) for csv_file in csv_files],
        ignore_index=True
    )
    
    print('Dopasowuję ścieżki do obrazów -', os.path.basename(__file__))
    # Automatyczne dopasowanie ścieżek do obrazów
    img_paths = all_data['image file path'].apply(
        lambda x: os.path.join(img_dir, x.split('/')[1], x.split('/')[-1].replace('.dcm', '.jpg'))
    )
    
    print('Pierwsze 5 ścieżek do obrazów:', img_paths.head().values)
    
    print('Przetwarzam obrazy -', os.path.basename(__file__))
    # Przetwarzanie obrazów
    images = []
    not_found = 0
    for img_path in img_paths:
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=img_size)  # Skaluje obraz do rozmiaru 512x512
            img_array = img_to_array(img) / 255.0  # Normalizacja do [0, 1]
            images.append(img_array)
        else:
            print(f"Plik obrazu nie znaleziony: {img_path}")
            not_found += 1
            images.append(np.zeros((*img_size, 3)))  # Wypełnia brakujące obrazy zerami
            
    print(f"Liczba brakujących obrazów: {not_found}")
    
    print('Konwertuję obrazy do formatu numpy -', os.path.basename(__file__))
    # Konwersja obrazów do formatu numpy
    X_image = np.array(images)
    
    print('Przetwarzam dane diagnostyczne -', os.path.basename(__file__))
    # Przetwarzanie danych diagnostycznych
    # (Sprawdzamy, czy każda kolumna istnieje, ponieważ różne CSV mogą zawierać różne kolumny)
    diagnostic_features = ['breast density', 'mass shape', 'mass margins', 'calc type', 'calc distribution']
    X_diagnostic = process_diagnostic_features(all_data, diagnostic_features)
    
    print('Generuję etykiety -', os.path.basename(__file__))
    # Generowanie etykiet (0 = BENIGN, 1 = MALIGNANT)
    y = all_data['pathology'].apply(lambda x: 1 if 'MALIGNANT' in x.upper() else 0).to_numpy()
    
    print('Zakończono wczytywanie i przetwarzanie danych -', os.path.basename(__file__))
    # Podział na zbiory treningowe i testowe
    X_image_train, X_image_test, X_diagnostic_train, X_diagnostic_test, y_train, y_test = train_test_split(
        X_image, X_diagnostic, y, test_size=test_size, random_state=42, stratify=y
    )

    return (X_image_train, X_diagnostic_train, y_train), (X_image_test, X_diagnostic_test, y_test)
