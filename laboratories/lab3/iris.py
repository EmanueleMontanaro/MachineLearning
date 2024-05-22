import numpy
import matplotlib.pyplot as plt
import pca
import lda
import commonResources
import classification

def conv(string):
    if 'Iris-setosa' in string:
        return 0
    elif 'Iris-versicolor' in string:
        return 1
    elif 'Iris-virginica' in string:
        return 2
    else:
        return string

def loadD():
    D = numpy.loadtxt('laboratories/lab1/iris.csv',usecols = range (0,4), delimiter=',',dtype=numpy.float32).T
    return D
    
def loadL():
    L = numpy.loadtxt('laboratories/lab1/iris.csv', usecols = 4, delimiter=',', dtype=numpy.float32 , converters={4: conv},encoding='utf-8')
    #usecols can be used with a range to select columns (last value excluded), converters is used to implement a custom dictionary
    #encode has to be used to treat string right, because dtype selects the type of value in the final result
    return L

def graphHist(D,L):
    D0 = D[:,L==0] #D0 is D with all rows but only with columns where L is equal to 0
    D1 = D[:, L==1]
    D2 = D[:, L==2]
    attributes = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }
    for index in range(4):
        plt.figure() #Create new figure
        plt.xlabel(attributes[index]) #Set label
        plt.hist(D0[index, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa') #Add variables to plot
        plt.hist(D1[index, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
        plt.hist(D2[index, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
        plt.legend() #Show automatic color legend
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('hist_%d.pdf' % index)
    plt.show()

def graphScatter(P,D,L):
    D0 = P.T @ D[:,L==0]
    D1 = P.T @ D[:, L==1]
    D2 = P.T @ D[:, L==2]

    attributes ={
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    for index1 in range(4):
        for index2 in range(4):
            if index1 == index2:
                continue
            plt.figure()
            plt.xlabel(attributes[index1])
            plt.ylabel(attributes[index2])
            plt.scatter(D0[index1, :], D0[index2, :], label = 'Setosa')
            plt.scatter(D1[index1, :], D1[index2, :], label = 'Versicolor')
            plt.scatter(D2[index1, :], D2[index2, :], label = 'Virginica')
            plt.legend()
            plt.tight_layout()
            #plt.savefig('scatter_%d%d.pdf' % index1 % index2)
        plt.show()

if __name__ =='__main__':
    D = loadD()
    L = loadL()
    classification.runClassifcation(D,L)
    plt.show()