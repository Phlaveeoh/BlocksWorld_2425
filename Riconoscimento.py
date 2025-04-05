import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()
dataD = digits.data
dataT = digits.target

XtrainD, xTestD, yTestD, yTrainD = train_test_split(dataD, dataT, test_size = 0.2, random_state=42)

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(XtrainD[i].reshape(8, 8), cmap='gray')
    plt.title(str(yTrainD[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()


numeroClassi = 10 

(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()
#print(type(xTrain))
#print(xTrain.shape)

#print(type(yTrain))
#print(yTrain.shape)

plt.figure(figsize=(10,10))
for i,immagini in enumerate(xTrain[0:25]):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(immagini, cmap=plt.cm.binary)
plt.show()



