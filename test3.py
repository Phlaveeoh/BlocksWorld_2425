from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Carica e prepara il dataset MNIST (28x28)
# -------------------------

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizzazione
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Rimodellamento: (batch, altezza, larghezza, canali)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# -------------------------
# Creazione e addestramento del modello MLP con i dati combinati
# -------------------------
""" model = Sequential([
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
    # Livello Dense con 256 neuroni
    Dense(256, activation='relu'),
    Dropout(0.5),
    # Livello Dense con 128 neuroni
    Dense(128, activation='relu'),
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
model.save("modelloDenso.h5")
# Salva il modello
model.save("modelloDenso.keras") """

model = load_model("BlocksWorld_2425\\modelloDenso.keras")

# Carica e prepara l'immagine grande (es. 224x224, grayscale)
image = cv2.imread('BlocksWorld_2425\\test_immagini\\scenaTelefono3.jpg')


# Converte l'immagine in scala di grigi
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)  # Inverte i colori per avere lo sfondo nero e le cifre bianche
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# Applica un leggero blur per ridurre il rumore
blurred = cv2.medianBlur(gray, 7)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# Define a kernel for dilation
dilated = cv2.erode(blurred, (1, 1), iterations=2)
cv2.imshow("Dilated", dilated)
cv2.waitKey(0)

# Applica la threshold per ottenere un'immagine binaria
# Utilizziamo THRESH_BINARY_INV per avere le cifre in bianco e lo sfondo in nero

thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow("Adaptive Threshold", thresh_adapt)
cv2.waitKey(0)

# Crea elemento structuring: ellisse di dimensione 19x19
kernelChiusura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
kernelApertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

opened = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernelApertura)
cv2.imshow("Opened", opened)
cv2.waitKey(0)

# Applica closing (dilate poi erode)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernelChiusura)

# Visualizza il risultato
cv2.imshow('Closed', closed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Usa connectedComponentsWithStats per segmentare l'immagine in componenti connesse
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Cicla su ogni contorno trovato
for cnt in contours:
    # Calcola il rettangolo di bounding per il contorno
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Filtra piccole regioni che potrebbero essere rumore
    if cv2.contourArea(cnt) < 50:
        continue

    # Estrai la ROI basata sul rettangolo di bounding
    roi = closed[y:y+h, x:x+w]
    #roi = cv2.bitwise_not(roi)
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
image_resized2 = cv2.resize(closed, (800, 800), interpolation=cv2.INTER_LINEAR)
cv2.imshow("Threshold", image_resized2)
cv2.imshow("Risultato", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()