from sklearn import datasets
import numpy as np
from adaline import AdalineSGD
from adaline import AdalineGD
import Perceptron
import random

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

def oitentaVintePerceptron(X, y):
    ###########################
    ### TREINA E CLASSIFICA ###
    slp = Perceptron()
    erros = 0

    Xtreina = X[0: int(len(X)*0.8)]
    ytreina = y[0: int(len(y)*0.8)]
    Xclassifica = X[int(len(X)*0.8): len(X)]
    yclassifica = y[int(len(y)*0.8): len(y)]

    slp.fit(Xtreina, ytreina)
    for i in range(len(Xclassifica)):
        yh = slp.predict(Xclassifica[i])
        erros += int(yclassifica[i] != yh)
    return 1 - (float(erros) / len(yclassifica))

def oitentaVinteGD(X, y):
    ###########################
    ### TREINA E CLASSIFICA ###
    slp = AdalineGD(eta=0.0001, n_iter=200)
    erros = 0

    Xtreina = X[0: int(len(X)*0.8)]
    ytreina = y[0: int(len(y)*0.8)]
    Xclassifica = X[int(len(X)*0.8): len(X)]
    yclassifica = y[int(len(y)*0.8): len(y)]

    slp.fit(Xtreina, ytreina)
    for i in range(len(Xclassifica)):
        yh = slp.predict(Xclassifica[i])
        erros += int(yclassifica[i] != yh)
    return 1 - (float(erros) / len(yclassifica))


def oitentaVinteSGD(X, y):
    ###########################
    ### TREINA E CLASSIFICA ###
    slp = AdalineSGD(eta=0.0001, n_iter=20)
    erros = 0

    Xtreina = X[0: int(len(X)*0.8)]
    ytreina = y[0: int(len(y)*0.8)]
    Xclassifica = X[int(len(X)*0.8): len(X)]
    yclassifica = y[int(len(y)*0.8): len(y)]

    slp.fit(Xtreina, ytreina)
    for i in range(len(Xclassifica)):
        yh = slp.predict(Xclassifica[i])
        erros += int(yclassifica[i] != yh)
    return 1 - (float(erros) / len(yclassifica))
#####################################
###### EXECUCAO DO ALGORITMO ########


digits = datasets.load_digits()
X = digits.data
y = np.where(digits.target==8, 1, -1)

print "Teste de dados Digitos com Gradiente Descendente"
for i in range(ITERACOES):

    Xn, Yn = aleatorio(X, y)
    erros = oitentaVinteGD(Xn, Yn)
    listaErros.append(erros)

result = np.mean(listaErros) * 100
print "%.4f" % result ,"%"

print "Teste de dados Digitos com Gradiente Stocastico"
for i in range(ITERACOES):

    Xn, Yn = aleatorio(X, y)
    erros = oitentaVinteSGD(Xn, Yn)
    listaErros.append(erros)

result = np.mean(listaErros) * 100
print "%.4f" % result ,"%"

print "Teste de dados Digitos com Perceptron"
for i in range(ITERACOES):
    Xn, Yn = aleatorio(X, y)
    erros = oitentaVinteSGD(Xn, Yn)
    listaErros.append(erros)

result = np.mean(listaErros) * 100
print "%.4f" % result ,"%"