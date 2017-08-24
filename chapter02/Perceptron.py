#encode utf-8

import numpy as np

#Classificador Perceptron
class Perceptron(object):
    #Hiperparâmetro
    alpha = 0.0
    nIter = 0
    
    #Parâmetros
    _w      = []
    _erros  = []
    
    #Métodos
    
    '''Construtor'''
    def __init__(self, alpha = 0.01, nIter = 10):
        self.alpha = alpha
        self.nIter = nIter
    
    '''Treinamento'''
    def treina(self, X, y):
            #X: Matriz com nAmostras linhas e nAtribultos colunas
            #y: Vetor com tamanho nAmostras contendo +1 ou -1 de acordo com a classe de cada amostra 
            self._w = np.zeros(X.shape[1]+1)
            self._erros = []
            for ep in range(self.nIter):
                erro = 0;
                for xi, yi in zip(X,y):
                    yh = self.classifica(xi)
                    e = yi - yh
                    self._w[1:] += self.alpha*e*xi
                    self._w[0] += self.alpha*e*1
                    erro += int(yi != yh)
                self._erros.append(erro)
            return self
    
    '''Calcula o z'''
    def _calcZ(self, X):
        return np.dot(X, self._w[1:]) + self._w[0]
    
    '''Classifica (calcula saída do classificador)'''
    def classifica(self, X):
        return np.where(self._calcZ(X) >= 0, 1, -1)
        
        
        
        
        
        
        
                 
                 
                 
                 
