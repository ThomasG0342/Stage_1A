from flask import Flask, render_template, request, redirect, url_for
import neurone
import reseaux_neurones as rneurones
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/plante_traitee", methods=["POST", "GET"])
def traitement_plante():
    if request.method == "POST":
        donnees = request.form
        hauteur = int(donnees.get('hauteur'))
        largeur = int(donnees.get('largeur'))
        
        W,b = neurone.artificial_neuron(neurone.X, neurone.y)
        
        new_plant = np.array([hauteur, largeur])
        prediction = neurone.predict(new_plant, W, b)
        
        neurone.graph_plante_toxique(new_plant, W, b)
        return render_template("plante.html", hauteur_html = hauteur, largeur_html = largeur, prediction_html = prediction[0][0], proba_html = prediction[1][0]*100)
    else:
        return redirect(url_for('plante'))
    
@app.route("/plante", methods=["POST", "GET"])
def plante():
    return render_template("plante.html")

@app.route("/annexe", methods=["POST", "GET"])
def annexe():
    return render_template("annexe.html")

@app.route("/reseaux_neurones", methods=["POST", "GET"])
def reseaux_neurones():
    return render_template("reseaux_neurones.html")

@app.route("/reseaux_neurone", methods=["POST", "GET"])
def traitement_reseaux_neurone():
    if request.method == "POST":
        donnees = request.form
        couche = int(donnees.get('couche'))
        neurone = int(donnees.get('neurone'))
              
        parametres = rneurones.deep_neural_network(couche, neurone, learning_rate=0.001, n_iter=1000)       
        
        
        return render_template("reseaux_neurones.html", couche_html=couche, neurone_html=neurone)
    else:
        return redirect(url_for('reseaux_neurones'))

if __name__ == '__main__':
    app.run(debug=True)