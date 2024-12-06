from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError
import os

def create_hybrid_model(image_shape=(512, 512, 3), diagnostic_shape=None, learning_rate=0.001):
    """
    Tworzy model hybrydowy łączący CNN dla obrazów i Dense dla danych diagnostycznych.

    Args:
        image_shape (tuple): Rozmiar wejścia obrazów (szerokość, wysokość, liczba kanałów).
        diagnostic_shape (int): Liczba cech diagnostycznych (obliczane dynamicznie, jeśli None).
        learning_rate (float): Współczynnik uczenia.

    Returns:
        Model: Skonstruowany model hybrydowy.
    """
    if diagnostic_shape is None:
        raise ValueError("diagnostic_shape cannot be None. Please provide the number of diagnostic features.")

    print('Zaczynam tworzyć model -', os.path.basename(__file__))
    # Ścieżka dla obrazów (CNN)
    image_input = Input(shape=image_shape, name='Image_Input')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Ścieżka dla danych diagnostycznych
    diagnostic_input = Input(shape=(diagnostic_shape,), name='Diagnostic_Input')
    y = Dense(16, activation='relu')(diagnostic_input)
    y = Dense(8, activation='relu')(y)

    # Połączenie ścieżek
    combined = concatenate([x, y], name='Concatenated_Features')
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(1, activation='sigmoid', name='Output')(z)
    
    print('Tworzę model -', os.path.basename(__file__))
    # Tworzenie modelu
    model = Model(inputs=[image_input, diagnostic_input], outputs=z, name='Hybrid_CNN_Diagnostic_Model')

    print('Kompiluję model -', os.path.basename(__file__))
    # Kompilacja modelu
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', MeanSquaredError()]
    )

    print('Zakończono tworzenie modelu -', os.path.basename(__file__))
    return model
