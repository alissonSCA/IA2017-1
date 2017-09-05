from sklearn import datasets
import numpy as np
from Perceptron import Perceptron
import random

QTDGRUPOS = 5
ITERACOES = 20
listaErros = []


def aleatorio(X, y):
    ##########################
    ###  RANDOM NOS DADOS  ###
    dados = []
    for i in range(len(y)):
        dados.append((X[i], y[i]))
    random.shuffle(dados)

    Xn = []
    yn = []
    for i in range(len(y)):
        Xn.append(dados[i][0])
        yn.append(dados[i][1])
    Xn = np.array(Xn)
    yn = np.array(yn)
    ###########################
    ###########################
    return Xn, yn


def defGrupos(X, y):
    ###########################
    #### SEPARAR OS GRUPOS ####
    dadosX = []
    dadosY = []
    init = 0
    final = len(y) / QTDGRUPOS

    for i in range(QTDGRUPOS):
        dadosX.append(X[init: final])
        dadosY.append(y[init: final])

        init += len(y) / QTDGRUPOS
        final += len(y) / QTDGRUPOS

    ###########################
    ###########################
    return dadosX, dadosY


def validacaoCruzada(X, y, tam):
    ###########################
    ### TREINA E CLASSIFICA ###
    slp = Perceptron()
    erros = 0
    treinoX = []
    treinoY = []

    for i in range(QTDGRUPOS):
        for j in range(QTDGRUPOS):
            if i <> j:
                if len(treinoX) == 0:
                    treinoX = X[j]
                    treinoY = y[j]
                else:
                    treinoX = np.concatenate([treinoX, X[j]])
                    treinoY = np.concatenate([treinoY, y[j]])

        slp.treina(treinoX, treinoY)
        for j in range(tam / QTDGRUPOS):
            yh = slp.classifica(X[i][j])
            erros += int(y[i][j] != yh)

    return 1 - ( float(erros) / (tam/QTDGRUPOS))

#####################################
###### EXECUCAO DO ALGORITMO ########
iris = datasets.load_iris()
X = iris.data
y = np.where(iris.target==0, 1, -1)

for i in range(ITERACOES):

    Xn, Yn = aleatorio(X, y)
    dadosX, dadosY = defGrupos(Xn, Yn)
    erros = validacaoCruzada(dadosX, dadosY, len(y))

    listaErros.append(erros)

result = np.mean(listaErros) * 100
print "%.4f" % result ,"%"
