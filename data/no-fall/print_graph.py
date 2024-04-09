import matplotlib.pyplot as plt
import pandas as pd
import sys
from urllib.parse import parse_qs
from datetime import datetime, timedelta

# Obtener el número de caída de los argumentos de la línea de comandos
fall_number = sys.argv[1]
path = "/home/naraujo/LPRO/Python-files/dataset/sin-caidas/" + fall_number + "/"

# Cargar los datos
df_gyro = pd.read_csv(path + 'gyro_data' + fall_number + '.csv', names=['time', 'x', 'y', 'z'], parse_dates=['time'])
df_accel = pd.read_csv(path + 'accel_data' + fall_number + '.csv', names=['time', 'x', 'y', 'z'], parse_dates=['time'])

# Establecer las marcas de tiempo como el índice
df_gyro.set_index('time', inplace=True)
df_accel.set_index('time', inplace=True)

# Graficar los datos
fig, axs = plt.subplots(2)
df_gyro.plot(ax=axs[0], title='Gyro Data')
df_accel.plot(ax=axs[1], title='Acceleration Data')
plt.show()