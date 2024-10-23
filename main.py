import tensorflow as tf  # Importa la biblioteca TensorFlow, utilizada para construir y entrenar modelos de aprendizaje automático.
import matplotlib.pyplot as plt  # Importa Matplotlib para crear visualizaciones gráficas.
import seaborn as sn  # Importa Seaborn, una biblioteca basada en Matplotlib que proporciona interfaces de alto nivel para visualización de datos.
import numpy as np  # Importa NumPy, que es fundamental para operaciones de álgebra lineal y manejo de matrices.
import pandas as pd  # Importa Pandas, que facilita la manipulación y análisis de datos, especialmente archivos CSV.
import math  # Importa la biblioteca Math, que proporciona funciones matemáticas.
import datetime  # Importa la biblioteca Datetime para trabajar con fechas y horas.
import platform  # Importa la biblioteca Platform para acceder a información del sistema operativo.
from sklearn.model_selection import train_test_split  # Importa la función para dividir datos en conjuntos de entrenamiento y validación.
from sklearn.manifold import TSNE  # Importa t-SNE, una técnica para la reducción de dimensionalidad y visualización de datos.

# Carga los archivos de datos de entrada desde el directorio de sólo lectura "../input/"
train = pd.read_csv('train.csv')  # Carga el conjunto de entrenamiento desde un archivo CSV.
test = pd.read_csv('test.csv')  # Carga el conjunto de prueba desde un archivo CSV.

# Las siguientes líneas están comentadas, pero servirían para mostrar las dimensiones de los conjuntos de datos.
"""
print('train:', train.shape)  # Imprime las dimensiones del conjunto de entrenamiento.
print('test:', test.shape)  # Imprime las dimensiones del conjunto de prueba.
"""

X = train.iloc[:, 1:785]  # Extrae las características (pixeles) del conjunto de entrenamiento.
y = train.iloc[:, 0]  # Extrae las etiquetas (números) del conjunto de entrenamiento.
X_test = test.iloc[:, 0:784]  # Extrae las características del conjunto de prueba.

# Las siguientes líneas están comentadas, pero servirían para visualizar la reducción de dimensionalidad de los datos con t-SNE.
"""
X_tsn = X / 255  # Normaliza los datos de entrada dividiendo por 255 (los valores de los pixeles).
tsne = TSNE()  # Inicializa el objeto t-SNE.
tsne_res = tsne.fit_transform(X_tsn)  # Aplica t-SNE para reducir la dimensionalidad de los datos.
plt.figure(figsize=(14, 12))  # Crea una figura con un tamaño específico para la visualización.
plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=y, s=2)  # Grafica los datos transformados.
plt.xticks([])  # Elimina las marcas del eje x.
plt.yticks([])  # Elimina las marcas del eje y.
plt.colorbar()  # Agrega una barra de color para indicar las clases.
plt.show()  # Muestra la visualización.
"""

# Divide el conjunto de entrenamiento en entrenamiento y validación, con un 20% para validación.
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1212)

# Las siguientes líneas están comentadas, pero servirían para mostrar las dimensiones de los conjuntos divididos.
"""
print('X_train:', X_train.shape)  # Imprime las dimensiones de las características de entrenamiento.
print('y_train:', y_train.shape)  # Imprime las dimensiones de las etiquetas de entrenamiento.
print('X_validation:', X_validation.shape)  # Imprime las dimensiones de las características de validación.
print('y_validation:', y_validation.shape)  # Imprime las dimensiones de las etiquetas de validación.
"""

# Reorganiza los datos para que tengan la forma (número de ejemplos, altura, ancho), convirtiendo a arreglos de NumPy.
x_train_re = X_train.to_numpy().reshape(33600, 28, 28)  # Reorganiza las imágenes de entrenamiento.
y_train_re = y_train.values  # Convierte las etiquetas de entrenamiento a un arreglo de NumPy.
x_validation_re = X_validation.to_numpy().reshape(8400, 28, 28)  # Reorganiza las imágenes de validación.
y_validation_re = y_validation.values  # Convierte las etiquetas de validación a un arreglo de NumPy.
x_test_re = test.to_numpy().reshape(28000, 28, 28)  # Reorganiza las imágenes de prueba.

# Guarda parámetros de imagen en constantes para su uso posterior en remodelado y entrenamiento.
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train_re.shape  # Obtiene el ancho y la altura de las imágenes.
IMAGE_CHANNELS = 1  # Define el número de canales de la imagen (1 para imágenes en escala de grises).
print('IMAGE_WIDTH:', IMAGE_WIDTH)  # Imprime el ancho de las imágenes.
print('IMAGE_HEIGHT:', IMAGE_HEIGHT)  # Imprime la altura de las imágenes.
print('IMAGE_CHANNELS:', IMAGE_CHANNELS)  # Imprime el número de canales de las imágenes.

pd.DataFrame(x_train_re[0])  # Crea un DataFrame de Pandas con la primera imagen del conjunto de entrenamiento.
plt.imshow(x_train_re[0], cmap=plt.cm.binary)  # Muestra la primera imagen en escala de grises.
plt.show()  # Muestra la visualización de la imagen.

# Muestra más ejemplos de entrenamiento para visualizar cómo se escribieron los dígitos.
numbers_to_display = 25  # Define cuántos ejemplos se mostrarán.
num_cells = math.ceil(math.sqrt(numbers_to_display))  # Calcula el número de celdas necesarias para la visualización.
plt.figure(figsize=(10, 10))  # Crea una figura con un tamaño específico para la visualización.
for i in range(numbers_to_display):  # Itera sobre el número de ejemplos a mostrar.
    plt.subplot(num_cells, num_cells, i + 1)  # Crea una subgráfica en la figura.
    plt.xticks([])  # Elimina las marcas del eje x.
    plt.yticks([])  # Elimina las marcas del eje y.
    plt.grid(False)  # Desactiva la cuadrícula.
    plt.imshow(x_train_re[i], cmap=plt.cm.binary)  # Muestra la imagen del dígito.
    plt.xlabel(y_train_re[i])  # Añade la etiqueta del dígito como título de la subgráfica.
plt.show()  # Muestra la visualización de los ejemplos de entrenamiento.
