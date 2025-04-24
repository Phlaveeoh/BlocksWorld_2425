from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Sequential
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np
# -------------------------
# Carica e prepara il dataset MNIST (28x28)
# -------------------------

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-elaborazione dei dati:
# 1. Normalizzazione: i pixel sono scalati tra 0 e 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. Rimodellamento: l'input deve avere la forma (altezza, larghezza, canali)
# Essendo MNIST in scala di grigi, il canale è 1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# -------------------------
# Creazione e addestramento del modello MLP con i dati combinati
# -------------------------
"""model = Sequential([
 # Primo livello convoluzionale
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    # Secondo livello convoluzionale
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    # Pooling per ridurre dimensionalità e complessità
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout per ridurre l'overfitting
    Dropout(0.25),
    
    # Appiattimento della mappa delle feature per poter collegare il fully connected
    Flatten(),
    # Livello Dense con 128 neuroni
    Dense(128, activation='relu'),
    Dropout(0.5),
    # Livello di output: 10 neuroni con attivazione softmax per la classificazione
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Visualizza il sommario del modello
model.summary()

# Addestramento del modello: 10 epoche e batch size di 128 (questi parametri possono essere ottimizzati)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Valutazione del modello sul test set
score = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# Salva il modello per poterlo successivamente caricare nella fase di inferenza
model.save("mnist_cnn.h5")
# Salva il modello
model.save("mlp_mnist.keras")"""

model = load_model("mlp_mnist.h5")

# Carica e prepara l'immagine grande (es. 224x224, grayscale)
image = cv2.imread('image.png')


# Converte l'immagine in scala di grigi
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applica un leggero blur per ridurre il rumore
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Applica la threshold per ottenere un'immagine binaria
# Utilizziamo THRESH_BINARY_INV per avere le cifre in bianco e lo sfondo in nero
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

# Usa connectedComponentsWithStats per segmentare l'immagine in componenti connesse
contours, hierarchy = cv2.findContours(thresh_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Cicla su ogni contorno trovato
for cnt in contours:
    # Calcola il rettangolo di bounding per il contorno
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Filtra piccole regioni che potrebbero essere rumore
    if cv2.contourArea(cnt) < 50:
        continue

    # Estrai la ROI basata sul rettangolo di bounding
    roi = thresh_adapt[y:y+h, x:x+w]
    roi_height, roi_width = roi.shape

    # Calcola un margine proporzionale (ad esempio, il 10% della dimensione minore)
    margin = int(0.2 * min(roi_height, roi_width))
    
    # Ritaglia la ROI usando il margine dinamico
    roi_cropped = roi[margin:roi_height-margin, margin:roi_width-margin]
    cv2.imshow("ROI cropped", roi_cropped)
    cv2.waitKey(0)
    # Ridimensiona la ROI ritagliata a 28x28 per il modello
    roi_resized = cv2.resize(roi_cropped, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalizza e adatta la dimensione per il modello
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_normalized = np.expand_dims(roi_normalized, axis=-3)
    roi_normalized = np.expand_dims(roi_normalized, axis=3)
    cv2.imshow("Final ROI", roi_resized)  # o la ROI dopo tutti i passaggi
    cv2.waitKey(0)

    # Esegui la previsione
    prediction = model.predict(roi_normalized)
    predicted_digit = np.argmax(prediction)
    print("AAAA",predicted_digit)

    # Disegna il rettangolo e la predizione sull'immagine originale
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, str(predicted_digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Visualizza l'immagine finale con i numeri riconosciuti e le relative posizioni
image_resized = cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)
cv2.imshow("Threshold", thresh)
cv2.imshow("Risultato", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()