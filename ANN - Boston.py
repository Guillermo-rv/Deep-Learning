# -*- coding: utf-8 -*-
"""
Created on Thu Jan 02 16:29:08 2024

@author: guill
"""


import pandas as pd


column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", 
    "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]


boston = pd.read_csv("housing.csv", sep="\s+", names=column_names, header=None)


print(boston.head())
print(boston.info())


boston = boston.astype(float)

print(boston.dtypes)

"""

CRIM  -- Tasa de criminalidad per cápita en la ciudad	
ZN	  -- Proporción de terrenos residenciales zonificados para lotes grandes (mayores a 25,000 pies cuadrados)	
INDUS -- Proporción de acres de negocios no minoristas por ciudad	float
CHAS  -- Variable ficticia del río Charles (1 si limita con el río, 0 en caso contrario)	
NOX	  -- Concentración de óxidos nítricos (partes por 10 millones)	float
RM	  -- Número promedio de habitaciones por vivienda	float
AGE	  -- Proporción de unidades ocupadas por propietarios construidas antes de 1940	float
DIS	  -- Distancias ponderadas a cinco centros de empleo en Boston	float
RAD   -- Índice de accesibilidad a autopistas radiales (número de carreteras principales cercanas)	int (puede tratarse como category)
TAX	  -- Tasa de impuesto a la propiedad por cada $10,000	float
PTRATIO	 -- Proporción alumno-profesor en cada ciudad	float
B	  -- Proporción de personas de ascendencia afroamericana en la ciudad (cálculo: 1000(Bk - 0.63)^2, donde Bk es la proporción de residentes afroamericanos)	float
LSTAT --Porcentaje de población de estatus socioeconómico bajo	float
MEDV  -- Valor medio de las viviendas ocupadas por propietarios en miles de dólares (Variable objetivo)

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X = boston.drop(columns=["MEDV"])  # Todas las columnas excepto el precio de la casa
y = boston["MEDV"]  # Variable objetivo (precio)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  


print("Media después de escalar:", np.mean(X_train, axis=0))
print("Desviación estándar después de escalar:", np.std(X_train, axis=0))


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa oculta 1
    Dense(32, activation='relu'),  # Capa oculta 2
    Dense(1, activation='linear')  # Capa de salida (valor continuo)
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)


test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"MAE en el conjunto de prueba: {test_mae:.2f}")


# Representación Grafica

import matplotlib.pyplot as plt


history_dict = history.history

# Graficar la pérdida (MSE) en entrenamiento y validación
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history_dict["loss"], label="Entrenamiento")
plt.plot(history_dict["val_loss"], label="Validación")
plt.xlabel("Épocas")
plt.ylabel("MSE (Error Cuadrático Medio)")
plt.title("Evolución del Error Durante el Entrenamiento")
plt.legend()

# Graficar el MAE en entrenamiento y validación
plt.subplot(1,2,2)
plt.plot(history_dict["mae"], label="Entrenamiento")
plt.plot(history_dict["val_mae"], label="Validación")
plt.xlabel("Épocas")
plt.ylabel("MAE (Error Absoluto Medio)")
plt.title("Evolución del MAE Durante el Entrenamiento")
plt.legend()

plt.show()



