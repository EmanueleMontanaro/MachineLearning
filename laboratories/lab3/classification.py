import pca
import lda
import numpy
import commonResources

#The objective is to separate the Versicolor and the Virginica classes.
#We want to preprocess data with PCA for applying LDA after. Therefore we want to check the goodness of our model with a validation dataset.
#To do so we will split the dataset: 2/3 will be training data, the remaining 1/3 will be validation data.

def load_iris(D,L): #Excludes Setosa class from our focus
    return D[:,L!=0],L[L!=0]

def split_db_2to1(D,L,seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL) #DTR and LTR are model training data and labels, DVAL and LVAL are validation data and labels

def computeThreshold(DTR,LTR):
    return (DTR[0,LTR==1].mean()+DTR[0,LTR==2].mean())/2.0

def runClassifcation(Dcomplete,Lcomplete):
    D, L = load_iris(Dcomplete,Lcomplete)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    lda.computeLDA(DTR,LTR,3,commonResources.computeN(DTR))
    lda.computeLDA(DVAL,LVAL,3,commonResources.computeN(DVAL))
    #For these to work I had to change the range of i in the lda.computeSw and lda.computeSb from k to 1,k then disable the scatter graph and change the bins of the hist from 10 to 5, then disable the Setosa row
    #As seen from the graphs, we could separate the two classes on the discriminant direction (1 since binary problem), therefore we can define a threshold which will be used by our model to label the unseen data
    #(data not present in the training data). Since we chose to have the Virginica with the highest mean (changeable with the sign of the LDA matrix), we will have elements on the right of the threshold labeled as
    #2 and elements on the left labeled as 1

    threshold = computeThreshold(DTR,LTR)

    #We now build an array of predicted labels

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL[0] >= threshold] = 2
    PVAL[DVAL[0] < threshold] = 1

    #PVAL is equal to LVAL but predicted and not given, therefore we will have disagreements on the overlapping sections
    #To check so, we count the numbers of disagreements
    count = 0
    for i in range(int(commonResources.computeN(DVAL))):
        if LVAL[i]!=PVAL[i]: count+=1
    print(count)