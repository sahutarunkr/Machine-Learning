def adaBoost(Wtemp, Xmatrix, Ymatrix):

    noOfData=len(Y)

    hmin = -1
    minError = 100000.0000
    for h in range(0, len(Xmatrix)):
        error = 0.000000000000001
        for m in range(0, noOfData):
            if (Ymatrix[h][m] != Y[m]):
                error = error + Wtemp[m]
        if (error < minError):
            minError = error
            hmin = h



    alphaTemp = 0.5 * (math.log((1 - minError) / minError))

    for m in range(0, noOfData):
        if (Ymatrix[hmin][m] != Y[m]):
            Wtemp[m] = Wtemp[m] * (math.exp(alphaTemp))
        else:
            Wtemp[m] = Wtemp[m] * (math.exp(-1*alphaTemp))
    alpha.append([alphaTemp,hmin,minError])
    for m in range(0, noOfData):
        if (Ymatrix[hmin][m] != Y[m]):
            W#print m,W[m]

    return Wtemp








from sklearn import tree
import numpy as np

from sklearn.metrics import accuracy_score
import math
data = (np.loadtxt("heart_train.data", dtype=int,delimiter=',', skiprows=0)).tolist()
#data = [[0, 2], [3, 4],[5,6]]


X = [x[1:] for x in data]
Y =[x[0] for x in data]
noOfData=len(Y);
for i in range(0,noOfData):
    if(Y[i]==0):
        Y[i]=-1

datatest = (np.loadtxt("heart_test.data", dtype=int,delimiter=',', skiprows=0)).tolist()
#data = [[0, 2], [3, 4],[5,6]]


Xtest = [x[1:] for x in datatest]
Ytest =[x[0] for x in datatest]
noOfDatatest=len(Ytest);
for i in range(0,noOfData):
    if(Ytest[i]==0):
        Ytest[i]=-1
totalNoOfAttributes=len(X)
M=3
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)


a = np.array([1, 2])
a.tolist()
z=0
Xmatrix=[]
Ymatrix=[]
XtestMatrix=[]
YtestMatrix=[]

al=0
a=0
W=[1/(float)(noOfData)]*(noOfData)
for i in range(0, len(X[0])):
    for j in range(i+1, len(X[0])):
        for k in range(j+1, len(X[0])):
            Xtemp=[[row[i],row[j],row[k]] for row in X]

            Xmatrix.append(Xtemp)
            Xtesttemp = [[row[i], row[j], row[k]] for row in Xtest]
            XtestMatrix.append(Xtesttemp)
            Ytemp=((clf.fit(Xtemp,Y)).predict(Xtemp)).tolist()
            Ymatrix.append(Ytemp)
            #print (a,i,j,k)

            #print a,i,j,k,accuracy_score(Y, Ytemp)
            a=a+1
accuracy=[]
iterationsNo=[]
for q in range(0,10):

    alpha = []
    W = [1 / (float)(noOfData)] * (noOfData)
    #print "dafasdfdf",W[30]
    for i in range(0, q):
        adaBoost(W, Xmatrix, Ymatrix)
    Yfinal = []
    #print alpha
    #print "adasfffffffffa",W[30]
    for i in range(0, len(Ytest)):
        k = 0.00000000000001
        for al in range(0, q):
            k = k + alpha[al][0] * ((clf.fit(Xmatrix[alpha[al][1]],Y)).predict(XtestMatrix[alpha[al][1]]).tolist())[i]
        Yfinal.append((int)(k / abs(k)))
    import matplotlib.pyplot as plt
    accuracy.append(accuracy_score(Ytest, Yfinal))
    iterationsNo.append(q+1)

    print  (accuracy_score(Ytest, Yfinal))
plt.plot(iterationsNo, accuracy, 'ro')
plt.axis([0, 11, 0, 1])

for a,b in zip(iterationsNo, accuracy):
    plt.text(a, b, str(b))
plt.show()














































