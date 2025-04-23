from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Lambda
)
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
mlp = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Addestramento
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp.fit(x_train, y_train, epochs=5)
mlp.save_weights("mlp.weights.h5")

def AdaptiveReshape(x):
    # usa tf.image.resize come layer Lambda
    return tf.image.resize(x, (28,28), method='bilinear')

#FIXME: reshape da 512x512 a 28x28, non ho trovato in cazzo di metodo adeguato
cnn = Sequential([
    Input(shape=(512, 512, 1)),
    Conv2D(16, 3, activation='relu', padding='same', strides=2),  # 256×256×16
    MaxPooling2D(2),                                              # 128×128×16
    Conv2D(32, 3, activation='relu', padding='same', strides=2),  #  64×64×32
    MaxPooling2D(2),                                              #  32×32×32
    Lambda(AdaptiveReshape),  # Porta a 28x28x32
    Conv2D(1, (1, 1), activation='linear'),  # (28, 28, 1)
    Flatten(),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

dummy_input = np.zeros((1, 512, 512, 1), dtype=np.float32)
cnn(dummy_input)  # Questo forza l'inizializzazione dei layer

mlp.load_weights("mlp.weights.h5")

# Copia i pesi dei layer Dense
cnn.layers[-3].set_weights(mlp.layers[-3].get_weights())  # Dense(512)
cnn.layers[-2].set_weights(mlp.layers[-2].get_weights())  # Dense(128)
cnn.layers[-1].set_weights(mlp.layers[-1].get_weights())  # Dense(10)

# Salva il modello
cnn.save("cnn_completo.h5")

#Carica il modello se già ce l'hai
cnn = load_model("BlocksWorld_2425\\cnn_completo.h5")

#-------------------------------------------------------------------------------------------------

def sliding_window_predict_large(scene_img, model, window_size=512, stride=50, threshold=0.8):
    # Converti in scala di grigi e in array
    scene_img = scene_img.convert("L")
    scene_array = np.array(scene_img)
    H, W = scene_array.shape

    predictions = []

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            window = scene_array[y:y+window_size, x:x+window_size]

            # Preprocessing
            window_input = np.expand_dims(window, axis=(0, -1))  # (1, 224, 224, 1)
            window_input = window_input.astype('float32') / 255.0

            pred = model.predict(window_input, verbose=0)
            confidence = np.max(pred)
            digit = np.argmax(pred)

            if confidence > threshold:
                predictions.append({
                    "digit": digit,
                    "confidence": float(confidence),
                    "position": (x, y)
                })

    return predictions

def draw_predictions(scene_img, predictions, window_size=224):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(scene_img.convert("L"), cmap='gray')

    for pred in predictions:
        x, y = pred["position"]
        digit = pred["digit"]
        conf = pred["confidence"]

        rect = patches.Rectangle((x, y), window_size, window_size, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + 5, y + 20, f'{digit} ({conf:.2f})', color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
from PIL import Image

# Carica l’immagine scattata col telefono
scene = Image.open("BlocksWorld_2425\\test_immagini\\scena.png")

# Fai predizioni
preds = sliding_window_predict_large(scene, cnn)

# Visualizza
draw_predictions(scene, preds)