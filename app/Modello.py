from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import visualkeras
from keras.utils import plot_model
import pandas  as pd
import seaborn as sn


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
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

# Visualizza il sommario del modello
model.summary()
visualkeras.layered_view(model, legend_text_spacing_offset=0).show() # display using your system viewer
visualkeras.layered_view(model, legend_text_spacing_offset=0, to_file='output.png') # write to disk

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Addestramento del modello: 10 epoche e batch size di 128 (questi parametri possono essere ottimizzati)
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Degine a subplot grid 1x2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

# Plot for accuracy and val_accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Accuracy', fontsize=13)
plt.ylim([0.0, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)

# Plot for loss and val_loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.ylim([0.0, 2])
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Valutazione del modello sul test set
score = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

y_pred = model.predict(x_test)
print("First image prediction prob:", y_pred[0])
print("Predictions:", y_pred.argmax(axis=1))

matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
print(matrix)

df_cm = pd.DataFrame(matrix, class_names, class_names)
plt.figure(figsize = (10,7))
sn.set_theme(font_scale=1.4) #for label size
sn.heatmap(df_cm, cmap="BuPu",annot=True, annot_kws={"size": 10}) #font size
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

# Salva il modello
model.save("modello.keras") 