import matplotlib.pyplot as plt
import math
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# -------------------------
# Carica e prepara il dataset MNIST (28x28)
# -------------------------
(x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.mnist.load_data()

# Visualizza alcune immagini ridimensionate
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train_raw[i], cmap=plt.cm.binary)
plt.show()

# Flatten dei dati MNIST ridimensionati (diventa un vettore di 64 elementi per immagine)
feature_vector_length = math.prod(x_train_raw.shape[1:])
x_train = x_train_raw.reshape(x_train_raw.shape[0], feature_vector_length)
x_test = x_test_raw.reshape(x_test_raw.shape[0], feature_vector_length)

# Normalizziamo il dataset MNIST.
scaler_mnist = MinMaxScaler()
x_train_norm = scaler_mnist.fit_transform(x_train)
x_test_norm = scaler_mnist.fit_transform(x_test)

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
mlp_combined.fit(x_train_norm, y_train)

# Valutazione del modello combinato
predictions = mlp_combined.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Cross-Validation
scores1 = cross_val_score(mlp_combined, x_test_norm, y_test, cv=10)
print("CV scores (MLP1):", scores1)
print("Mean accuracy value (MLP1):", scores1.mean())

mlp_combined.save("mlp.keras")

# CNN model definition
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    #Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten()
])
# Model architecture visualization
model.summary()
