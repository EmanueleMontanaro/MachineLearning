import scipy.linalg
import numpy
import matplotlib.pyplot as plt
import commonResources

def computeSw(D,L,k,N): #Function to compute the within class matrix as a weighted sum of the covariance matrices of each class
    Sw = 0
    for i in range(1,k):
        Swc = computeSWc(splitClasses(D,L,i))
        nc = commonResources.computeN(splitClasses(D,L,i))
        Sw += (Swc * nc)
    return Sw/N

def splitClasses(D,L,i):
     return D[:, L==i]

def computeSWc(Dclass): #Function to compute the covariance matrice of a single class
    mu = commonResources.computeMu(Dclass)
    DC = commonResources.computeDC(Dclass,mu)
    SWc = commonResources.computeC(DC,commonResources.computeN(Dclass))
    return SWc

def computeSb(D,L,k,N):
    Sb = 0
    for i in range(1,k):
        Sb+=commonResources.computeN(splitClasses(D,L,i)) * (classCenteredMean(D,splitClasses(D,L,i)) @ classCenteredMean(D,splitClasses(D,L,i)).T)
    return Sb/N

def classCenteredMean(D,Dclass):
    return (commonResources.computeMu(Dclass) - commonResources.computeMu(D))

def genEigProblem(Sw,Sb,m):
    s, U = scipy.linalg.eigh(Sb,Sw) #Solves the generalized eigenvalue probelm for hermitian matrices
    W = U[:, ::-1][:, 0:m] #Columns of W are not necessarily orthogonal, we can find a basis U for the subspace spanned by W using the SVD of W
    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:m]
    return U

def genEigProblemByJoint(Sw, Sb, m):
    #Step 1: estimating matrix P1 such that the within class covariance of the transformed points P1x is the identity. Can be done through SVD of Sw with some edits

    U, s, _ = numpy.linalg.svd(Sw) #This computes U Epsilon U.T but we need Epsilon^-1/2

    #s is 1-D diagonal array containing the diagonal of Epsilon, we can compute Epsilon^-1/2 like 1.0/s**0.5
    #By exploiting broadcasting we can compute P1 like follows

    P1 = numpy.dot(U * commonResources.vrow(1.0/(s**0.5)), U.T) #or P1 = numpy.dot(numpy.dot(U, numpy.diag(1.0/s**0.5))), U.T) to build a diagonal matrix from a 1-D array.

    #Step 2: compute the transformed Sb (Sbt)

    Sbt = P1 @ Sb @ P1.T

    #Step 3: compute P2 with the same steps for P1 but on Sbt

    U2, s2, _ = numpy.linalg.svd(Sbt)
    P2 = U2[:, 0:m]
    return numpy.dot(P2.T,P1).T

def graphScatter(U,D,L):
    D0 = U.T @ D[:,L==0]
    D1 = U.T @ D[:, L==1]
    D2 = U.T @ D[:, L==2]
    plt.figure()
    plt.scatter(D0[0, :], D0[1, :], label = 'Setosa')
    plt.scatter(D1[0, :], D1[1, :], label = 'Versicolor')
    plt.scatter(D2[0, :], D2[1, :], label = 'Virginica')
    plt.title('LDA')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('scatter_%d%d.pdf' % index1 % index2)

def graphHist(U,D,L):
    #D0 = U.T @ D[:,L==0]
    D1 = U.T @ D[:, L==1]
    D2 = U.T @ D[:, L==2]  
    plt.figure() #Create new figure
    #plt.hist(D0[0, :], bins = 5, density = True, alpha = 0.4, label = 'Setosa') #Add variables to plot
    plt.hist(D1[0, :], bins = 5, density = True, alpha = 0.4, label = 'Versicolor')
    plt.hist(D2[0, :], bins = 5, density = True, alpha = 0.4, label = 'Virginica')
    plt.title('LDA')
    plt.legend() #Show automatic color legend
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure

def computeLDA(D,L,k,N):
    Sw = computeSw(D,L,k,N)
    Sb = computeSb(D,L,k,N)
    U = genEigProblemByJoint(Sw, Sb, 2)
    #graphScatter(U,D,L)
    graphHist(U,D,L)