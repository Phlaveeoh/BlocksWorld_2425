from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from skimage.util import view_as_windows
from PIL import Image
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
# -------------------------
# Carica e prepara il dataset MNIST (28x28)
# -------------------------

(x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------
# Creazione e addestramento del modello MLP con i dati combinati
# -------------------------
"""mlp = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Salva il modello
mlp.save("mlp_mnist.h5")"""

mlp = load_model("mlp_mnist.h5")

# Carica e prepara l'immagine grande (es. 224x224, grayscale)
img = Image.open("image copy.png").convert('L').resize((224, 224))
img = np.array(img) / 255.0  # normalizza

# Impostazioni per patch e stride
patch_size = 28
stride = 14  # sovrapposizione

# Scorri l'immagine e estrai le patch
patches_list = []
coordinates = []

for y in range(0, img.shape[0] - patch_size + 1, stride):
    for x in range(0, img.shape[1] - patch_size + 1, stride):
        patch = img[y:y+patch_size, x:x+patch_size]
        patches_list.append(patch)
        coordinates.append((x, y))

# Predizione su tutte le patch
predictions = mlp.predict(np.array(patches_list))  # Modifica con il tuo MLP
confidences = np.max(predictions, axis=1)  # Ottieni la confidenza di ogni predizione

# Estrai la classe predetta (indice della probabilità massima)
predicted_classes = np.argmax(predictions, axis=1)

# Soglia di confidenza (ad esempio, scarta predizioni con confidenza bassa)
threshold = 0.9  # soglia da regolare in base alla tua esigenza
predictions_filtered = [
    predicted_classes[i] if confidences[i] > threshold else -1
    for i in range(len(predicted_classes))
]

# Visualizzazione delle predizioni sulla griglia
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')

# Disegna le finestre (patches) e le predizioni sopra l'immagine
for i, (x, y) in enumerate(coordinates):
    # Se la predizione è valida (diversa da -1), disegna il rettangolo
    if predictions_filtered[i] != -1:
        rect = patches.Rectangle((x, y), patch_size, patch_size, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-2, str(predictions_filtered[i]), color='red', fontsize=8)

plt.show()
