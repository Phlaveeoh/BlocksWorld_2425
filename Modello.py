from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizzazione
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Rimodellamento: (batch, altezza, larghezza, canali)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


model = Sequential([
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

# Salva il modello
model.save("DENSONE.keras") 