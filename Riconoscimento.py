import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.transform import resize
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# -------------------------
# Carica e prepara il dataset digits (8x8)
# -------------------------
digits = load_digits()
data_digits = digits.data    # shape: (n_samples, 64)
target_digits = digits.target

# Normalizziamo il dataset digits (i valori sono tipicamente nell'intervallo 0-16)
scaler_digits = MinMaxScaler()
data_digits_norm = scaler_digits.fit_transform(data_digits)

# Suddividiamo il dataset digits in train e test (se necessario)
Xtrain_digits, Xtest_digits, ytrain_digits, ytest_digits = train_test_split(
    data_digits_norm, target_digits, test_size=0.2, random_state=42
)

# -------------------------
# Carica e prepara il dataset MNIST (28x28)
# -------------------------
(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()

# Ridimensiona le immagini MNIST da 28x28 a 8x8 utilizzando l'interpolazione.
# Convertiamo le immagini in float32 e ridimensioniamo ogni immagine.
xTrain_down = np.array([resize(image.astype(np.float32), (8, 8), anti_aliasing=True)
                        for image in xTrain])
xTest_down = np.array([resize(image.astype(np.float32), (8, 8), anti_aliasing=True)
                       for image in xTest])

# Visualizza alcune immagini ridimensionate
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(xTrain_down[i], cmap=plt.cm.binary)
plt.show()

# Flatten dei dati MNIST ridimensionati (diventa un vettore di 64 elementi per immagine)
xTrain_flat = xTrain_down.reshape(xTrain_down.shape[0], -1)
xTest_flat = xTest_down.reshape(xTest_down.shape[0], -1)

# Normalizziamo il dataset MNIST.
scaler_mnist = MinMaxScaler()
xTrain_norm = scaler_mnist.fit_transform(xTrain_flat)
xTest_norm = scaler_mnist.transform(xTest_flat)

# -------------------------
# Combina i dataset digits e MNIST
# -------------------------
# Utilizziamo il training set di MNIST e quello di digits
combined_data = np.vstack((Xtrain_digits, xTrain_norm))
combined_target = np.concatenate((ytrain_digits, yTrain))

# Suddividiamo il dataset combinato in train e test
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    combined_data, combined_target, test_size=0.2, random_state=42
)

# ACCURATEZZA IN BASE ALLE ITERAZIONI:
# 75  = 96.77%
# 100 = 96.81%
# 125 = 96.93%
# 126 = 96.97%
# 137 = 96.89%
# 150 = 96.88%

# -------------------------
# Creazione e addestramento del modello MLP con i dati combinati
# -------------------------
mlp_combined = MLPClassifier(
    solver='adam',
    activation='relu',
    hidden_layer_sizes=(512,128),
    max_iter=126,
    random_state=1,
    verbose=True
)
mlp_combined.fit(X_train_combined, y_train_combined)

# Valutazione del modello combinato
predictions = mlp_combined.predict(X_test_combined)
accuracy = metrics.accuracy_score(y_test_combined, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
