import pandas as pd
import os

# Define la ruta base donde se encuentran las carpetas 'fall' y 'no-fall'
base_path = '/mnt/c/Users/juanp/OneDrive - Universidade de Vigo/Escritorio/ML/data'

# Define las subcarpetas a procesar
subfolders = ['fall', 'no-fall']

for subfolder in subfolders:
    folder_path = os.path.join(base_path, subfolder)
    # Itera a través de cada subcarpeta dentro de 'fall' y 'no-fall'
    for sub_dir in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, sub_dir)):
            accel_file_path = os.path.join(folder_path, sub_dir, f'accel_data{sub_dir}.csv')
            gyro_file_path = os.path.join(folder_path, sub_dir, f'gyro_data{sub_dir}.csv')
            output_file_path = os.path.join(folder_path, sub_dir, f'data{sub_dir}.csv')
            
            # Leer los archivos CSV
            accel_data = pd.read_csv(accel_file_path, header=None, usecols=[1, 2, 3])
            gyro_data = pd.read_csv(gyro_file_path, header=None, usecols=[1, 2, 3])
            
            # Concatena los datos del acelerómetro y giroscopio horizontalmente
            combined_data = pd.concat([accel_data, gyro_data], axis=1)
            
            # Guarda el archivo combinado
            combined_data.to_csv(output_file_path, index=False, header=False)
