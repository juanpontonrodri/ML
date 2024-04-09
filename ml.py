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
    
    categories = ['fall', 'no-fall']
    for category in categories:
        category_path = os.path.join(base_path, category)
        print(f"Procesando categoría: {category}, en ruta: {category_path}")
        
        if not os.path.exists(category_path):
            print(f"La ruta {category_path} no existe.")
            continue
        
        for folder_name in os.listdir(category_path):
            folder_path = os.path.join(category_path, folder_name)
            csv_file = os.path.join(folder_path, f'data{folder_name}.csv')
            
            if not os.path.isfile(csv_file):
                print(f"Archivo no encontrado: {csv_file}")
                continue
            
            data = pd.read_csv(csv_file, header=None)
            # Asumiendo que se necesitan todas las filas y las últimas 6 columnas
            data = data.iloc[:, -6:].to_numpy()
            
            if len(data) > max_seq_length:
                data = data[:max_seq_length, :]
            elif len(data) < max_seq_length:
                padding = max_seq_length - len(data)
                data = np.pad(data, ((0, padding), (0, 0)), 'constant', constant_values=0)
            
            X.append(data)
            y.append(1 if category == 'fall' else 0)
                
    print(f"Datos cargados: {len(X)} muestras.")
    return np.array(X), np.array(y)

def preprocess_data(X, y):
    # Aplanar los datos para la estandarización
    X_flattened = X.reshape((X.shape[0], -1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flattened)
    # Volver a dar forma a los datos para que sean adecuados para el modelo CNN
    X_processed = X_scaled.reshape((X.shape[0], 300, 6))  # Ajustar según la forma de tus datos
    y_encoded = to_categorical(y)
    return X_processed, y_encoded, scaler

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

def predict_from_csv(file_path, model, scaler):
    # Cargar los datos
    data = pd.read_csv(file_path, header=None)
    # Preprocesar los datos como se hizo durante el entrenamiento
    data = data.iloc[:, -6:].to_numpy()  # Asumiendo que se necesitan las últimas 6 columnas
    
    if len(data) > 300:
        data = data[:300, :]
    elif len(data) < 300:
        padding = 300 - len(data)
        data = np.pad(data, ((0, padding), (0, 0)), 'constant', constant_values=0)
    
    data_flattened = data.flatten().reshape(1, -1)
    data_scaled = scaler.transform(data_flattened)
    data_scaled = data_scaled.reshape((1, 300, 6))
    
    # Hacer la predicción
    prediction = model.predict(data_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    
    return "Caída" if predicted_class[0] == 1 else "No caída"

# Cargar y preparar el dataset
base_path = '/home/juan/ML/data'
X, y = load_dataset(base_path)
X_processed, y_encoded, scaler = preprocess_data(X, y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Construir y entrenar el modelo
input_shape = (300, 6)  # Ajustar según la forma de tus datos
model = build_model(input_shape)
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluar el modelo
model.evaluate(X_test, y_test)

model.save('modelo.h5')  # Guarda el modelo

# Uso de la función de predicción con un archivo nuevo
file_path = '/home/juan/ML/data/no-fall/2/data2.csv'  # Actualiza esto a la ruta de tu archivo CSV
prediction = predict_from_csv(file_path, model, scaler)
print(prediction)
