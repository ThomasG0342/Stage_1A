import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import os

X,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
X = X - X.min(axis=0)
y = y.reshape((y.shape[0], 1))

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    print(A)
    return (A >= 0.5, A)

def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X)

    Loss = []

    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    #y_pred = predict(X, W, b)
    #print(accuracy_score(y, y_pred))

    #plt.plot(Loss)
    #plt.show()

    return (W, b)
    

W, b = artificial_neuron(X, y)

def graph_plante_toxique(new_plant, W, b):
    
    x0 = np.linspace(-1, 12, 100)
    x1 = (-W[0] * x0 - b ) / W[1]

    plt.scatter(X[:, 0],X[:, 1], c=y, cmap='summer')
    plt.scatter(new_plant[0], new_plant[1], c='red')
    plt.scatter(X[y[:,0]==0, 0], X[y[:,0]==0, 1], c='green', label='plante non toxique')
    plt.scatter(X[y[:,0]==1, 0], X[y[:,0]==1, 1], c='yellow', label='plante toxique')
    plt.plot(x0, x1, c='orange', lw = 3, label = 'Frontière : sépare toxique et non-toxique')
    plt.legend()
    plt.ylabel('Largeur de la plante')
    plt.xlabel('Hauteur de la plante')
    
    nom_fichier = "graphique_test_plante"
    dossier = "./static/img"

    chemin_complet = os.path.join(dossier, nom_fichier + ".png")

    # Sauvegarde du graphique
    plt.savefig(chemin_complet, dpi=300, bbox_inches='tight')
    plt.close()
    return 0
