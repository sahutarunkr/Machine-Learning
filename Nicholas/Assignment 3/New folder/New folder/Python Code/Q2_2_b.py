def coordinateDescenet(Xmatrix, Ymatrix, i):

    nsum = 0
    dsum = 0.000000001
    for m in range(0, len(Y)):

        if (Ymatrix[i][m] == Y[m]):
            sum1 = 0
            for j in range(0, len(alphaM)):
                if (j != i):
                    sum1 = sum1 + alphaM[j] * Ymatrix[j][m]
            try:
                nsum = nsum + math.exp(-1 * Y[m] * sum1)
            except OverflowError:
                k = (-1 * Y[m] * sum1) / abs(-1 * Y[m] * sum1)
                nsum = nsum + k * float('inf')
        if (Ymatrix[i][m] != Y[m]):
            sum2 = 0
            for j in range(0, len(alphaM)):
                if (j != i):
                    sum2 = sum2 + alphaM[j] * Ymatrix[j][m]

            try:
                dsum = dsum + math.exp(-1 * Y[m] * sum2)
            except OverflowError:
                k = (-1 * Y[m] * sum2) / abs(-1 * Y[m] * sum2)
                dsum = dsum + k * float('inf')
    logx = nsum / dsum

    alphaM[i] = 0.5 * math.log(logx)


    #print alphaM
    return alphaM



from sklearn import tree
import numpy as np

from sklearn.metrics import accuracy_score
import math
from random import randint
datatest = (np.loadtxt("heart_test.data", dtype=int,delimiter=',', skiprows=0)).tolist()
#data = [[0, 2], [3, 4],[5,6]]


Xtest = [x[1:] for x in datatest]
Ytest =[x[0] for x in datatest]
for i in range(0,len(Ytest)):
    if(Ytest[i]==0):
        Ytest[i]=-1


data = (np.loadtxt("heart_train.data", dtype=int,delimiter=',', skiprows=0)).tolist()
#data = [[0, 2], [3, 4],[5,6]]


X = [x[1:] for x in data]
Y =[x[0] for x in data]
noOfData=len(Y);
for i in range(0,noOfData):
    if(Y[i]==0):
        Y[i]=-1
totalNoOfAttributes=len(X)
M=3
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
hforoneatt=(((len(Y)+1)*2-2))

a = np.array([1, 2])
a.tolist()
z=0
Xmatrix=[]
Ymatrix=[]
XtestMatrix=[]
YtestMatrix=[]
Yoneheight=[]

for x in range(0,len(Y)):
    xtemp=[-1]*x + [1]*(len(Y)-x)
    Yoneheight.append(xtemp)

for x in range(0, len(Y)):
    ytemp = [1] * x + [-1] * (len(Y) - x)
    Yoneheight.append(ytemp)



al=0
a=0
W=[1/(float)(noOfData)]*(noOfData)
for k in range(0, len(X[0])):
    Xtemp = [[row[k]] for row in X]
    Xmatrix.append(Xtemp)
    Xtesttemp = [[row[k]] for row in Xtest]
    XtestMatrix.append(Xtesttemp)
    Ytemp = ((clf.fit(Xtemp, Y)).predict(Xtemp)).tolist()

    Ymatrix.append(Ytemp)#=[[-1,1,1,1,1],[-1,-1,1,1,1],[-1,-1,-1,1,1],[-1,-1,-1,-1,1]]

#print len(Ymatrix[0])


alphaM=[0]*(len(Ymatrix))
#print alphaM
q=10000
for round in range(0,q):
    a = randint(0, len(alphaM)-1)

    alphaM = coordinateDescenet( Xmatrix, Ymatrix,a)

    Yfinal = []
    Eloss=0
    for m in range(0,len(Y)):
        sum=0
        for t in range(0,len(alphaM)):
            sum=sum+alphaM[t]*Ymatrix[t][m]
            #print(sum)



        Eloss = Eloss + (math.exp(-1 * sum * Y[m]))




    #print  (a,Eloss,alphaM)

#Yfinal=[0]*len(Ytest)
for i in range(0, len(Ytest)):
    k = 0.0000000000001
    for al in range(0, len(alphaM)):
        k = k + alphaM[al] * ((clf.fit(Xmatrix[al], Y)).predict(XtestMatrix[al]))[i]
    Yfinal.append(int(k / abs(k)))
print  ( accuracy_score(Ytest, Yfinal))
