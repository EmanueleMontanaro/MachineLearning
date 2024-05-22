import numpy
import matplotlib.pyplot as plt

def loadD():
    D = numpy.loadtxt("project/trainData.txt",usecols = range (0,6), delimiter=',',dtype=numpy.float32).T
    return D

def loadL():
    L = numpy.loadtxt("project/trainData.txt",usecols = 6, delimiter=',',dtype=numpy.int32)
    return L

def histGraph(D,L):
    D0 = D[:,L==0] 
    D1 = D[:, L==1]
    for index in range(6):
        plt.figure()
        plt.hist(D0[index,:],bins = 10, density=True, alpha=0.4, label='True')
        plt.hist(D1[index,:],bins = 10, density=True, alpha=0.4, label='False')
        plt.legend()
        plt.tight_layout()
    plt.show()

def scatterGraph(D,L):
    D0 = D[:,L==0] 
    D1 = D[:, L==1]
    for index in range(3):
        plt.figure()
        plt.scatter(D0[index,:],D0[index+1,:], label='True')
        plt.scatter(D1[index,:],D1[index+1,:], label='False')
        plt.legend()
        plt.tight_layout()
    plt.show()

scatterGraph(loadD(),loadL())