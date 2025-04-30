import Riconoscimento as ric
import Problema as problema
from keras.models import load_model
## carichiamo le 2 iimmagini
## funzione riconoscimento.py che di darà la matrice 6x6
## salviamo le matrici
## le diamo in pasto a problema.py
## creiamo le gif 
pathImgInput = ".\\test_immagini\\scenaTelefono3.jpg"
pathImgOutput = ".\\test_immagini\\scenaTelefono5.jpg"
pathModello = ".\\modelloDenso.keras"

modello = load_model(pathModello)

matriceInput = ric.riconosci_immagine(pathImgInput, modello)
matriceOutput = ric.riconosci_immagine(pathImgOutput, modello)

print("Matrice di input:")
print(matriceInput)
print("Matrice di output:")
print(matriceOutput)

problemone = problema.BlocksWorldProblem(problema.Board(matriceInput), problema.Board(matriceOutput))
problema.execute("Da scena 3 a 5", problema.aStar, problemone)