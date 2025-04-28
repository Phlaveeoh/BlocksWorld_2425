from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
 
morph_size = 0
max_operator = 4
max_elem = 2
max_kernel_size = 21
title_trackbar_operator_type = 'Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat'
title_trackbar_element_type = 'Element:\n 0: Rect - 1: Cross - 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n + 1'
title_window = 'Morphology Transformations Demo'
morph_op_dic = {0: cv.MORPH_OPEN, 1: cv.MORPH_CLOSE, 2: cv.MORPH_GRADIENT, 3: cv.MORPH_TOPHAT, 4: cv.MORPH_BLACKHAT}

# Carica e prepara l'immagine grande (es. 224x224, grayscale)
image = cv.imread('BlocksWorld_2425\\test_immagini\\scenaTelefono3.jpg')


# Converte l'immagine in scala di grigi
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.bitwise_not(gray)  # Inverte i colori per avere lo sfondo nero e le cifre bianche
cv.imshow("Gray", gray)
cv.waitKey(0)

# Applica un leggero blur per ridurre il rumore
blurred = cv.medianBlur(gray, 9)
cv.imshow("Blurred", blurred)
cv.waitKey(0)

# Define a kernel for dilation
dilated = cv.erode(blurred, (1, 1), iterations=2)
cv.imshow("Dilated", dilated)
cv.waitKey(0)

# Applica la threshold per ottenere un'immagine binaria
# Utilizziamo THRESH_BINARY_INV per avere le cifre in bianco e lo sfondo in nero
_, thresh = cv.threshold(dilated, 200, 255, cv.THRESH_BINARY_INV)

thresh_adapt = cv.adaptiveThreshold(dilated, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv.THRESH_BINARY_INV, 11, 2)
 
def morphology_operations(val):
    morph_operator = cv.getTrackbarPos(title_trackbar_operator_type, title_window)
    morph_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_window)
    morph_elem = 0
    val_type = cv.getTrackbarPos(title_trackbar_element_type, title_window)
    if val_type == 0:
        morph_elem = cv.MORPH_RECT
    elif val_type == 1:
        morph_elem = cv.MORPH_CROSS
    elif val_type == 2:
        morph_elem = cv.MORPH_ELLIPSE
 
    element = cv.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
    operation = morph_op_dic[morph_operator]
    dst = cv.morphologyEx(src, operation, element)
    cv.imshow(title_window, dst)
 
parser = argparse.ArgumentParser(description='Code for More Morphology Transformations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='LinuxLogo.jpg')
args = parser.parse_args()
 
src = thresh_adapt
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
 
cv.namedWindow(title_window)
cv.createTrackbar(title_trackbar_operator_type, title_window , 0, max_operator, morphology_operations)
cv.createTrackbar(title_trackbar_element_type, title_window , 0, max_elem, morphology_operations)
cv.createTrackbar(title_trackbar_kernel_size, title_window , 0, max_kernel_size, morphology_operations)
 
morphology_operations(0)
cv.waitKey()