import cv2
import numpy as np

# Con get_contour_depth posso calcolare la profondità del contorno che sto analizzando
def get_contour_depth(hierarchy, idx):
    depth = 0
    parent = hierarchy[idx][3]
    while parent != -1:
        depth += 1
        parent = hierarchy[parent][3]
    return depth

# controllare se ci sono solamente 6 numeri
def sborra(numeri):
    if len(numeri) > 6:
        exit()
    if 7 in numeri or 8 in numeri or 9 in numeri:
        for n in range(len(numeri)):
            if numeri[n] == 7 or numeri[n] == 8 or numeri[n] == 9:
                numeri[n] = 1
    # prendizioni controlliamo se ci sono doppioni
    doppioni = []
    for n in range(len(numeri)):
        if numeri[n] not in doppioni:
            doppioni.append(numeri[n])
        else:
            for d in range(1,7):
                if d not in numeri:
                    numeri[n] = d

#Funzione che riconosce i numeri in un'immagine e restituisce una lista di tuple (numero, x, y)
def riconosci_immagine(percorsoImmagine, model):
    '''Funzione che riconosce i numeri in un'immagine e restituisce una lista di tuple (numero, x, y).\n
    Richiede come argomento il percorso dell'immagine e il modello CNN per il riconoscimento dei numeri.'''
    # -------------------------
    # Preprocessing dell'immagine
    # -------------------------
    # Carico l'immagine
    # Vecchio: immagine = cv2.imread('BlocksWorld_2425\\test_immagini\\scenaTelefono5.jpg')
    immagine = cv2.imread(percorsoImmagine)

    # Converto l'immagine in scala di grigi
    gray = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)  # Inverte i colori per avere lo sfondo nero e le cifre bianche
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

    # Applico un leggero blur per ridurre il rumore
    blurred = cv2.medianBlur(gray, 7)
    cv2.imshow("Blurred", blurred)
    cv2.waitKey(0)

    # Dilato l'immagine per inspessire le cifre
    dilated = cv2.erode(blurred, (3, 3), iterations=2)
    cv2.imshow("Dilated", dilated)
    cv2.waitKey(0)

    # Applico una threshold adattiva per ottenere un'immagine binaria
    thresholded = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow("Adaptive Threshold", thresholded)
    cv2.waitKey(0)

    # Applico apertura per rimuovere piccoli rumori (erode poi dilate)
    kernelApertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernelApertura)
    cv2.imshow("Opened", opened)
    cv2.waitKey(0)

    # Applico closing per riempire i contorni vuoti (dilate poi erode)
    kernelChiusura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernelChiusura)
    cv2.imshow('Closed', closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # -------------------------
    # Riconoscimento dei numeri
    # -------------------------
    # Ora trovo i contorni nell'immagine
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]  # shape (N, 4)

    # Prendo solo i contorni a profondità 2 (cioè i contorni dei numeri)
    depth2_contours = []
    for i, cnt in enumerate(contours):
        if get_contour_depth(hierarchy, i) == 2:
            depth2_contours.append(cnt)
    
    numero = []
    xMio = []
    yMio = []

    # Lavoro su ogni contorno delle cifre che ho trovato
    for cnt in depth2_contours:
        
        #Calcolo il rettangolo del contorno
        x, y, w, h = cv2.boundingRect(cnt)
        
        # se l'are del contorno è molto piccola la scartoperchè probabilmente è un rumore residuo
        if cv2.contourArea(cnt) < 50:
            continue
        
        # Estraggo la ROI (Region of Interest)
        roi = closed[y:y+h, x:x+w]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        
        # Aggiungo un bordo alla ROI per migliorare la previsione
        roi_bordered = cv2.copyMakeBorder(roi, top=30, bottom=30, left=30, right=30, borderType=cv2.BORDER_CONSTANT, value=0)
        cv2.imshow("ROI con bordo", roi_bordered)
        cv2.waitKey(0)
        
        # Ridimensiono la ROI a 28x28 pixel (dimensione del dataset MNIST)
        roi_resized = cv2.resize(roi_bordered, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imshow("ROI", roi_resized)
        cv2.waitKey(0)
        
        # Normalizzo l'immagine per il modello di predizione(valori tra 0 e 1)
        roi_normalized = roi_resized.astype("float32") / 255.0
        roi_normalized = np.expand_dims(roi_normalized, axis=-3)
        roi_normalized = np.expand_dims(roi_normalized, axis=3)

        # Eseguo la previsione usando il modello
        prediction = model.predict(roi_normalized)
        predicted_digit = np.argmax(prediction)
        print("Cifra Predetta:",predicted_digit)
        
        # Salvo il numero che ho trovato e le sue coordinate
        numero.append(predicted_digit)
        print(f"Numero trovato {predicted_digit}")
        xMio.append(x)
        yMio.append(y)
        print(numero)
        print(xMio)
        print(yMio)
        
        # Disegno un rettangolo attorno alla cifra trovata e scrivo il numero predetto
        cv2.rectangle(immagine, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(immagine, str(predicted_digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostro l'immagine dopo tutte le trasformazioni morfologiche e le predizioni fatte su di essa attraverso il modello
    # Le ridimensiono per renderle più leggibili
    image_resized = cv2.resize(immagine, (800, 800), interpolation=cv2.INTER_LINEAR)
    image_resized2 = cv2.resize(closed, (800, 800), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Threshold", image_resized2)
    cv2.imshow("Risultato", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # -------------------------
    # Operazioni finali per togliere doppioni e numeri non validi
    # -------------------------
    sborra(numero, xMio, yMio)
    return list(zip(numero,xMio,yMio))