from data_preparation import load_and_prepare_data
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Ścieżki do danych i modelu
IMG_DIR = "data/jpeg"
CSV_DIR = "data/csv"
MODEL_PATH_FULL = "models/model_full.keras"
MODEL_PATH_EARLY = "models/model_early.keras"

# Wczytanie danych
print("Wczytywanie danych...")
(train_X_image, train_X_diagnostic, train_y), (test_X_image, test_X_diagnostic, test_y) = load_and_prepare_data(
    img_dir=IMG_DIR,
    csv_dir=CSV_DIR,
    img_size=(512, 512)
)

# Wczytanie modeli
print("Wczytywanie modeli...")
model_full = load_model(MODEL_PATH_FULL)
model_early = load_model(MODEL_PATH_EARLY)

model_full.summary()
model_early.summary()

# Przewidywania na zbiorze testowym
print("Przewidywanie...")
predictions_full = model_full.predict([test_X_image, test_X_diagnostic])
predictions_early = model_early.predict([test_X_image, test_X_diagnostic])

# Binarna klasyfikacja z progiem 0.5
predictions_full_binary = (predictions_full > 0.5).astype(int)
predictions_early_binary = (predictions_early > 0.5).astype(int)

# Funkcja oceny modeli
def evaluate_model(name, true_labels, predictions, predictions_binary):
    print(f"\n=== Wyniki dla {name} ===")
    accuracy = accuracy_score(true_labels, predictions_binary)
    precision = precision_score(true_labels, predictions_binary)
    recall = recall_score(true_labels, predictions_binary)
    f1 = f1_score(true_labels, predictions_binary)
    mse = mean_squared_error(true_labels, predictions)
    print(f"Dokładność: {accuracy:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    print("\nMetryki klasyfikacji:")
    print(classification_report(true_labels, predictions_binary, target_names=["Benign", "Malignant"]))
    
    # Macierz pomyłek
    cm = confusion_matrix(true_labels, predictions_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Przewidywana klasa")
    plt.ylabel("Rzeczywista klasa")
    plt.title(f"Macierz pomyłek - {name}")
    plt.show()

# Ocena modelu z pełnymi epokami
evaluate_model("Model z pełnymi epokami", test_y, predictions_full, predictions_full_binary)

# Ocena modelu z EarlyStopping
evaluate_model("Model z EarlyStopping", test_y, predictions_early, predictions_early_binary)
