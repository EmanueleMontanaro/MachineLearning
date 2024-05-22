import iris
import numpy
import matplotlib.pyplot as plt
import commonResources

def computeEig(C): #s are eigenvalues, sorted from highest to lowest, U columns are eigenvectors
    s, U = numpy.linalg.eigh(C)
    return s, U

def computeEigSVD(C): #same as above but eigenvalues are from lowest to highest, U columns are already inverted
    U, s, Vh = numpy.linalg.svd(C)
    return U, s, Vh

def computeP(U,m):
    return U[:, 0:m]

def graphScatter(P,D,L):
    D0 = P.T @ D[:,L==0]
    D1 = P.T @ D[:, L==1]
    D2 = P.T @ D[:, L==2]
    plt.figure()
    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')
    plt.title('PCA')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('scatter_%d%d.pdf' % index1 % index2)

def graphHist(P,D,L):
    D0 = P.T @ D[:,L==0]
    D1 = P.T @ D[:, L==1]
    D2 = P.T @ D[:, L==2]    
    plt.figure() #Create new figure
    plt.hist(D0[0, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa') #Add variables to plot
    plt.hist(D1[0, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
    plt.hist(D2[0, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
    plt.title('PCA')
    plt.legend() #Show automatic color legend
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure

def computePCA(D,L):
    mu = commonResources.vcol(commonResources.computeMu(D))
    N = commonResources.computeN(D)
    DC = commonResources.computeDC(D,mu)
    C = commonResources.computeC(DC,N)
    U, s, Vh = computeEigSVD(C)
    P = computeP(U,4)
    graphScatter(P,D,L)
    graphHist(P,D,L)