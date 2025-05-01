from flask import Flask, request, render_template, url_for
import os
from keras.models import load_model
import Riconoscimento as ric
import Problema as problema
import CreaGif

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), '../static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static/result'

# Carica il modello una sola volta
modello = load_model("models\\modelloDenso.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    gif_url = None

    if request.method == "POST":
        input_file = request.files["input_img"]
        output_file = request.files["output_img"]

        if input_file and output_file:
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.jpg")
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
            
            input_file.save(input_path)
            output_file.save(output_path)

            # Riconoscimento
            matriceInput = ric.riconosci_immagine(input_path, modello)
            matriceOutput = ric.riconosci_immagine(output_path, modello)

            print("Matrice di input:", matriceInput)
            print("Matrice di output:", matriceOutput)

            # Risoluzione problema e creazione GIF
            problemone = problema.BlocksWorldProblem(problema.Board(matriceInput), problema.Board(matriceOutput))
            soluzione = problema.execute("Soluzione del problema", problema.aStar, problemone)
            CreaGif.create(matriceInput, soluzione)
            gif_url = url_for('static', filename='result/BlocksWorld_Solution.gif')

    return render_template("index.html", gif_url=gif_url)

if __name__ == "__main__":
    app.run(debug=True)
