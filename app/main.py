import eventlet
eventlet.monkey_patch()  # DEVE ESSERE la primissima istruzione, prima di qualsiasi altro import

import os
from flask import Flask, request, render_template, url_for
from flask_socketio import SocketIO
from keras.models import load_model
import Riconoscimento as ric
import Problema as problema
import CreaGif
import time

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), '../static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static/result'
app.config['SERVER_NAME'] = 'localhost:5000'
app.config['PREFERRED_URL_SCHEME'] = 'http'

# Inizializza SocketIO
socketio = SocketIO(app)

# Carica il modello una sola volta (assicurati che il path sia corretto)
modello = load_model("models\\modelloDenso.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_file = request.files["input_img"]
        output_file = request.files["output_img"]

        if input_file and output_file:
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg")
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
            
            input_file.save(input_path)
            output_file.save(output_path)

            # Avvia il task in background per evitare di bloccare il thread della richiesta
            socketio.start_background_task(process_images, input_path, output_path)

    # Renderizza la pagina iniziale: il client continuerà ad ascoltare gli eventi WS
    return render_template("index.html")

def process_images(input_path, output_path):
    # Invia aggiornamento: Inizia la soluzione
    socketio.emit('status', {'msg': 'Cercando una soluzione...'})
    socketio.sleep(0)

    start_time = time.time()

    # Elaborazione delle immagini
    matriceInput = ric.riconosci_immagine(input_path, modello)
    matriceOutput = ric.riconosci_immagine(output_path, modello)
    
    # Definisci e risolvi il problema
    problemone = problema.BlocksWorldProblem(problema.Board(matriceInput), problema.Board(matriceOutput))
    soluzione = problema.execute("Soluzione del problema", problema.aStar, problemone)
    
    # Calcola il tempo impiegato
    solution_time = time.time() - start_time
    socketio.emit('status', {'msg': f'Soluzione trovata in {solution_time:.2f} secondi.'})
    socketio.emit('status', {'msg': 'Generando la GIF con la soluzione...'})
    socketio.sleep(0)
    
    # Creazione della GIF
    start_gif_time = time.time()
    CreaGif.create(matriceInput, soluzione)

    # Calcola il tempo per la generazione della GIF
    gif_time = time.time() - start_gif_time
    socketio.emit('status', {'msg': f'GIF generata in {gif_time:.2f} secondi.'})
    socketio.sleep(0)
    
    # Per chiamare url_for che necessita del contesto, crea un app context
    with app.app_context():
        gif_url = url_for('static', filename='result/BlocksWorld_Solution.gif', _external=True)
    
    # Invia il messaggio finale con l'URL della GIF pronta
    socketio.emit('gif_ready', {'url': gif_url})


if __name__ == '__main__':
    socketio.run(app, debug=True)
