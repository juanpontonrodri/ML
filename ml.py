import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_dataset(base_path):
    X, y = [], []
    max_seq_length = 300  # Número fijo de pasos de tiempo
    
    # Asumiendo que el resto de tu función load_dataset permanece igual
    
    # Después de cargar los datos pero antes de convertirlos a np.array
    # Aplica truncamiento o relleno
    X_processed = []
    for seq in X:
        if len(seq) > max_seq_length:
            # Truncar las secuencias más largas
            new_seq = seq[:max_seq_length]
        else:
            # Rellenar las secuencias más cortas con ceros
            new_seq = np.pad(seq, ((0, max_seq_length - len(seq)), (0, 0)), 'constant')
        X_processed.append(new_seq)
    
    print(f"Datos cargados: {len(X)} muestras.")
    return np.array(X), np.array(y)


def preprocess_data(X, y):
    # Aplanar los datos ya que cada observación está en una matriz 2D
    X_flattened = X.reshape((X.shape[0], -1))
    
    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flattened)
    
    # Volver a dar forma a los datos para que sean adecuados para el modelo CNN
    X_processed = X_scaled.reshape((X.shape[0], X.shape[1], X.shape[2]))
    
    # Codificar las etiquetas
    y_encoded = to_categorical(y)
    
    return X_processed, y_encoded

def build_model(input_shape):
    model = Sequential([
        Conv1D(64, 2, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(128, 2, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cargar y preparar el dataset
base_path = '/mnt/c/Users/juanp/OneDrive - Universidade de Vigo/Escritorio/ML/data'
X, y = load_dataset(base_path)
X_processed, y_encoded = preprocess_data(X, y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Construir y entrenar el modelo
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape)
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluar el modelo
model.evaluate(X_test, y_test)
