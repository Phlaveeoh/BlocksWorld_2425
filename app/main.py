import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, request, render_template, url_for
from flask_socketio import SocketIO
from keras.models import load_model
import Riconoscimento as ric
import Problema as problema
from GifCreator import GifCreator
import time
from collections import Counter
import MatrixMapper as mm

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), '../static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = os.path.join('static', 'result')
app.config['SERVER_NAME'] = 'localhost:5000'
app.config['PREFERRED_URL_SCHEME'] = 'http'

socketio = SocketIO(app, cors_allowed_origins="*")
modello = load_model(os.path.join('models', 'modelloIntelligente.keras'))
 
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_file = request.files["input_img"]
        output_file = request.files["output_img"]
        heuristic = request.form.get("heuristic", "blocked")  # Nuovo parametro

        if input_file and output_file:
            # Aggiungiamo l'euristica al nome del file per evitare conflitti
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{heuristic}.jpg")
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{heuristic}.jpg")
            
            input_file.save(input_path)
            output_file.save(output_path)

            # Passiamo l'euristica al task in background
            socketio.start_background_task(process_images, input_path, output_path, heuristic)

    return render_template("index.html")

def process_images(input_path, output_path, heuristic):
    # Invia aggiornamento iniziale
    socketio.emit('status', {'msg': f'[{heuristic}] Cercando una soluzione...', 'heuristic': heuristic})
    socketio.sleep(0)

    start_time = time.time()

    # Elaborazione delle immagini
    tuplaInput = ric.riconosci_immagine(input_path, modello)
    tuplaOutput = ric.riconosci_immagine(output_path, modello)

    numeriInput = [num for num, _, _ in tuplaInput]
    numeriOutput = [num for num, _, _ in tuplaOutput]

    if Counter(numeriInput) == Counter(numeriOutput):
        socketio.emit('status', {'msg': f'[{heuristic}] I numeri coincidono. Procedo...', 'heuristic': heuristic})
        socketio.sleep(0)
    else:
        socketio.emit('status', {'msg': f'[{heuristic}] ERRORE: I numeri NON coincidono!', 'heuristic': heuristic, 'type': 'error'})
        socketio.sleep(0)
        return

    matriceInput = mm.digitalizza(tuplaInput)
    matriceOutput = mm.digitalizza(tuplaOutput)

    # Definisci e risolvi il problema
    problemone = problema.BlocksWorldProblem(problema.Board(matriceInput), problema.Board(matriceOutput))
    soluzione = problema.execute("Soluzione del problema", problema.aStar, problemone, heuristic=heuristic)

    # Tempo totale di soluzione
    solution_time = time.time() - start_time
    socketio.emit('status', {'msg': f'[{heuristic}] Soluzione trovata in {solution_time:.2f} secondi.', 'heuristic': heuristic})
    socketio.sleep(0)
    socketio.emit('status', {'msg': f'[{heuristic}] Generando la GIF...', 'heuristic': heuristic})
    socketio.sleep(0)

    # Creazione GIF
    start_gif_time = time.time()
    mosse_soluzione, explored_nodes, frontier_nodes, execution_time = soluzione
    gifCreator = GifCreator(matriceInput, mosse_soluzione)
    percorsoGif = gifCreator.create()
    gif_time = time.time() - start_gif_time

    socketio.emit('status', {'msg': f'[{heuristic}] GIF generata in {gif_time:.2f} secondi.', 'heuristic': heuristic})
    socketio.emit('status', {'msg': f'[{heuristic}] GIF salvata in "{percorsoGif}".', 'heuristic': heuristic})
    socketio.sleep(0)

    # Invia URL GIF e statistiche al client
    with app.app_context():
        gif_url = url_for('static', filename=percorsoGif, _external=True)

    socketio.emit('gif_ready', {
        'url': gif_url,
        'heuristic': heuristic,
        'stats': {
            'visitedNodes': explored_nodes,
            'executionTime': round(execution_time * 1000, 2),
            'pathCost': len(mosse_soluzione)
        }
    })



if __name__ == '__main__':
    # Crea la cartella uploads se non esiste
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    socketio.run(app, debug=True)