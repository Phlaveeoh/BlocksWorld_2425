import Riconoscimento as ric
import Problema as problema
from keras.models import load_model
## carichiamo le 2 iimmagini
## funzione riconoscimento.py che di darà la matrice 6x6
## salviamo le matrici
## le diamo in pasto a problema.py
## crediamo le gif 
pathImgInput = ".\\test_immagini\\scenaTelefono4.jpg"
pathImgOutput = ".\\test_immagini\\scenaTelefono4.jpg"
pathModello = "\\modelloDenso.keras"

modello = load_model(pathModello)

matriceInput = ric.riconosci_Immagine(pathImgInput, modello)
matriceInput = ric.riconosci_Immagine(pathImgOutput, modello)

problemone = problema.BlocksWorldProblem(problema.Board(matriceInput), problema.Board(matriceInput))
problema.execute(problemone)

