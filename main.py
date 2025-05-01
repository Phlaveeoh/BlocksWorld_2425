import Riconoscimento as ric
import Problema as problema
from keras.models import load_model
import CreaGif

pathImgInput = ".\\immagini\\scenaTelefono3.jpg"
pathImgOutput = ".\\immagini\\scenaTelefono5.jpg"
pathModello = ".\\modelli_addestrati\\modelloDenso.keras"

# Carico il modello pre-addestrato
modello = load_model(pathModello)

# Riconosco i numeri nelle immagini
matriceInput = ric.riconosci_immagine(pathImgInput, modello)
matriceOutput = ric.riconosci_immagine(pathImgOutput, modello)

# Stampo le matrici per verificare il risultato
print("Matrice di input:")
print(matriceInput)
print("Matrice di output:")
print(matriceOutput)

# Costruisco il problema
problemone = problema.BlocksWorldProblem(problema.Board(matriceInput), problema.Board(matriceOutput))
# Eseguo l'algoritmo A* per trovare il percorso ottimale
soluzione = problema.execute("Da scena 3 a 5", problema.aStar, problemone)

CreaGif.create(matriceInput, soluzione)

# Creo la gif finale