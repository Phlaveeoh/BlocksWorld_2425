from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

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
model = load_model("Progettone\\BlocksWorld_2425\\modelloDenso.keras")
# -------------------------
# Caricamento e pre-processing dell'immagine grande
# -------------------------

# Carica l'immagine
immagine = cv2.imread('Progettone\\BlocksWorld_2425\\test_immagini\\scenaTelefono3.jpg')

# Converte l'immagine in scala di grigi
gray = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)  # Inverte i colori per avere lo sfondo nero e le cifre bianche
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# Applica un leggero blur per ridurre il rumore
blurred = cv2.medianBlur(gray, 7)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# Dilato l'immagine per inspessire le cifre
dilated = cv2.erode(blurred, (3, 3), iterations=2)
cv2.imshow("Dilated", dilated)
cv2.waitKey(0)

# Applica una threshold adattiva per ottenere un'immagine binaria
thresholded = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow("Adaptive Threshold", thresholded)
cv2.waitKey(0)

# Applica apertura per rimuovere piccoli rumori (erode poi dilate)
kernelApertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernelApertura)
cv2.imshow("Opened", opened)
cv2.waitKey(0)

# Applica closing per riempire i contorni vuoti (dilate poi erode)
kernelChiusura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
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

    # Esegui la previsione
    prediction = model.predict(roi_normalized)
    predicted_digit = np.argmax(prediction)
    print("Cifra Predetta:",predicted_digit)

    # Disegna il rettangolo e la predizione sull'immagine originale
    cv2.rectangle(immagine, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(immagine, str(predicted_digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Visualizza l'immagine finale con i numeri riconosciuti e le relative posizioni
image_resized = cv2.resize(immagine, (800, 800), interpolation=cv2.INTER_LINEAR)
image_resized2 = cv2.resize(closed, (800, 800), interpolation=cv2.INTER_LINEAR)
cv2.imshow("Threshold", image_resized2)
cv2.imshow("Risultato", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

