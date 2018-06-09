def adaBoost(Wtemp, Xmatrix, Ymatrix):
    noOfData=len(Y)
    h_min = -1
    min_error = 100000.0000
    for h in range(0, len(Xmatrix)):
        error = 0.00000000000001
        for k in range(0, noOfData):
            if Ymatrix[h][k] != Y[k]:
                error = error + Wtemp[k]
        #print(h,error,Wtemp)
        if error < min_error:
            min_error = error
            h_min = h
    in_log = ((1-min_error)/min_error)
    alpha_temp = 0.5*(math.log(in_log))
    for m in range(0,noOfData):
        Wtemp[m]=Wtemp[m]*(math.exp(-Y[m]*Ymatrix[h_min][m]*alpha_temp))/(2*math.pow((1-min_error)*min_error,0.5))
    alpha.append([alpha_temp, h_min, min_error])

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
totalNoOfAttributes=len(X)
M=3
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)


a = np.array([1, 2])
a.tolist()
z=0
Xmatrix=[]
Ymatrix=[]

al=0
a=0
W=[1/(float)(noOfData)]*(noOfData)
for i in range(0, len(X[0])):
    for j in range(i+1, len(X[0])):
        for k in range(j+1, len(X[0])):
            Xtemp=[[row[i],row[j],row[k]] for row in X]
            Xmatrix.append(Xtemp)
            Ytemp=((clf.fit(Xtemp,Y)).predict(Xtemp)).tolist()
            Ymatrix.append(Ytemp)
            #print (a,i,j,k,accuracy_score(Y, Ytemp))
            a=a+1

for q in range(3,4):
    alpha = []
    W = [1 / (float)(noOfData)] * (noOfData)

    for i in range(0, q):
        adaBoost(W, Xmatrix, Ymatrix)

        Eloss = 0
        for m in range(0, len(Y)):
            sum = 0
            for t in range(0, len(alpha)):
                sum = sum + alpha[t][0] * Ymatrix[alpha[t][1]][m]
                # print(sum)

            Eloss = Eloss + (math.exp(-1 * sum * Y[m]))
        print(Eloss,alpha)
    Yfinal = []
    #print (alpha)


    #print "adasfffffffffa",W[30]
    for i in range(0, noOfData):
        k = 0.00000000000001
        for al in range(0, q):
            clf.fit(Xmatrix[alpha[al][1]], Y).predict(Xmatrix[alpha[al][1]])
            strin = "decisiontree" + str(alpha[al][1]) + ".dot"
            tree.export_graphviz(clf, out_file=strin)
            k = k + alpha[al][0] * Ymatrix[alpha[al][1]][i]
        Yfinal.append((int)(k / abs(k)))

    print  (accuracy_score(Y, Yfinal))

