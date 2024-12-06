from data_preparation import load_and_prepare_data
from model import create_hybrid_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Ścieżki do katalogów
IMG_DIR = "data/jpeg"
CSV_DIR = "data/csv"

print("Zaczynam wczytywać dane -", os.path.basename(__file__))
# Wczytanie danych
(train_X_image, train_X_diagnostic, train_y), (test_X_image, test_X_diagnostic, test_y) = load_and_prepare_data(
    img_dir=IMG_DIR, 
    csv_dir=CSV_DIR, 
    img_size=(512, 512)
)

print("Zakończono wczytywanie danych -", os.path.basename(__file__))

print("Zaczynam tworzyć model z pełnymi epokami -", os.path.basename(__file__))
# Model z pełnymi epokami
model_full = create_hybrid_model(image_shape=(512, 512, 3), diagnostic_shape=train_X_diagnostic.shape[1])
history_full = model_full.fit(
    [train_X_image, train_X_diagnostic],  # Przekazanie dwóch wejść
    train_y,  # Etykiety wyjściowe
    epochs=50,
    batch_size=16,
    validation_data=([test_X_image, test_X_diagnostic], test_y)  # Walidacja
)
print("Zakończono tworzenie modelu z pełnymi epokami -", os.path.basename(__file__))
print("Zaczynam tworzyć model z EarlyStopping -", os.path.basename(__file__))
# Model z EarlyStopping
model_early = create_hybrid_model(image_shape=(512, 512, 3), diagnostic_shape=train_X_diagnostic.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history_early = model_early.fit(
    [train_X_image, train_X_diagnostic],  # Przekazanie dwóch wejść
    train_y,  # Etykiety wyjściowe
    epochs=50,
    batch_size=16,
    validation_data=([test_X_image, test_X_diagnostic], test_y),  # Walidacja
    callbacks=[early_stopping]
)
print("Zakończono tworzenie modelu z EarlyStopping -", os.path.basename(__file__))

# Binary Crossentropy Loss w czasie
plt.figure(figsize=(10, 6))
plt.plot(history_full.history['loss'], label='Pełne epoki - trening (loss)')
plt.plot(history_full.history['val_loss'], label='Pełne epoki - walidacja (val_loss)')
plt.xlabel('Epoki')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.title('Porównanie Binary Crossentropy Loss na przestrzeni epok - pełne epoki')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history_early.history['loss'], label='EarlyStopping - trening (loss)')
plt.plot(history_early.history['val_loss'], label='EarlyStopping - walidacja (val_loss)')
plt.xlabel('Epoki')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.title('Porównanie Binary Crossentropy Loss na przestrzeni epok - EarlyStopping')
plt.show()


# Mean Squared Error (MSE) w czasie
plt.figure(figsize=(10, 6))
plt.plot(history_full.history['mean_squared_error'], label='Pełne epoki - trening (MSE)')
plt.plot(history_full.history['val_mean_squared_error'], label='Pełne epoki - walidacja (MSE)')
plt.xlabel('Epoki')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.title('Porównanie MSE na przestrzeni epok - pełne epoki')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history_early.history['mean_squared_error'], label='EarlyStopping - trening (MSE)')
plt.plot(history_early.history['val_mean_squared_error'], label='EarlyStopping - walidacja (MSE)')
plt.xlabel('Epoki')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.title('Porównanie MSE na przestrzeni epok - EarlyStopping')
plt.show()

# Dokładność (Accuracy) w czasie
plt.figure(figsize=(10, 6))
plt.plot(history_full.history['accuracy'], label='Pełne epoki - trening (accuracy)')
plt.plot(history_full.history['val_accuracy'], label='Pełne epoki - walidacja (val_accuracy)')
plt.xlabel('Epoki')
plt.ylabel('Dokładność (Accuracy)')
plt.legend()
plt.title('Porównanie dokładności walidacyjnej na przestrzeni epok - pełne epoki')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history_early.history['accuracy'], label='EarlyStopping - trening (accuracy)')
plt.plot(history_early.history['val_accuracy'], label='EarlyStopping - walidacja (val_accuracy)')
plt.xlabel('Epoki')
plt.ylabel('Dokładność (Accuracy)')
plt.legend()
plt.title('Porównanie dokładności walidacyjnej na przestrzeni epok - EarlyStopping')
plt.show()

train_full_classification_error = [1 - acc for acc in history_full.history['accuracy']]
test_full_classification_error = [1 - acc for acc in history_full.history['val_accuracy']]
train_early_classification_error = [1 - acc for acc in history_early.history['accuracy']]
test_early_classification_error = [1 - acc for acc in history_early.history['val_accuracy']]

# Błąd klasyfikacji w czasie
plt.figure(figsize=(10, 6))
plt.plot(train_full_classification_error, label='Pełne epoki - trening')
plt.plot(test_full_classification_error, label='Pełne epoki - walidacja')
plt.xlabel('Epoki')
plt.ylabel('Błąd klasyfikacji')
plt.legend()
plt.title('Porównanie błędu klasyfikacji na przestrzeni epok - pełne epoki')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_early_classification_error, label='EarlyStopping - trening')
plt.plot(test_early_classification_error, label='EarlyStopping - walidacja')
plt.xlabel('Epoki')
plt.ylabel('Błąd klasyfikacji')
plt.legend()
plt.title('Porównanie błędu klasyfikacji na przestrzeni epok - EarlyStopping')
plt.show()

model_full.save("model_full_cnn.keras")
model_early.save("model_early_cnn.keras")

# Zapis modelu
choice = input("Czy zapisać model? (t/n): ")
if choice.lower() == 't':
    model_full.save("models/model_full_cnn.keras")
    model_early.save("models/model_early_cnn.keras")
    print("Modele zapisane pomyślnie.")
else:
    print("Modele nie zostały zapisane.")
    
print("Zakończono pomyślnie -", os.path.basename(__file__))
