from sklearn import datasets
import numpy as np
from Perceptron import Perceptron

iris = datasets.load_iris()
X = iris.data
y = np.where(iris.target==0, 1, -1)

slp = Perceptron()
slp.treina(X,y)

erros = 0
for xi, yi in zip(X,y):
    yh = slp.classifica(xi)
    erros += int(yi != yh)
print erros    
