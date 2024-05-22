def computeMu(D): #Dataset mean
    return vcol(D.mean(1))

def vcol(x): #Function to reshape a 1-D vector as a column vector
    return x.reshape((x.size,1))

def vrow(x): #Function to reshape a 1-D vector as a row vector
    return x.reshape((1,x.size))

def computeN(D): #N is the number of samples, can be obtained as follows
    return float(D.shape[1])

def computeDC(D,mu): #DC is the Dataset Centered matrix, meaning the whole dataset matrix - the dataset mean
    return (D-vcol(mu))

def computeC(DC,N): #C is the covariance matrix
    return (DC @ DC.T)/N