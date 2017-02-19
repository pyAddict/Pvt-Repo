import numpy as np
import pandas as pd
from sklearn.cross_validation import  train_test_split
data=np.genfromtxt('exp2d1.txt',delimiter=',')
temp=pd.DataFrame(data)
#data visualization
temp.head()
x=data[:,0:2]
y=data[:,np.newaxis,2]
y=y.ravel()
x_train,x_test,y_train,y_test=train_test_split(x,y)
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.lda import LDA
ldd=LDA()
#ldd=LinearDiscriminantAnalysis()
ldd.fit(x_train,y_train)
y_pred=ldd.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))
ldd.intercept_
#plotting
w = ldd.coef_[0] ;
a = -w[0] / w[1] ;
x1=x_test[:,0];x2=x_test[:,1];
y1=a * x1 - (ldd.intercept_[0]) / w[1]
import matplotlib.pyplot as plt
for i in range(0,len(y_test)):
    c=['red' if y_test[i]==1 else 'blue']
    #lb=['x1' if y_test[i]==1 else 'x2']
    plt.scatter(x1[i],x2[i],color=c) 
    plt.hold(True)
plt.plot(x1,y1, 'k-',label='Decision_Line')
plt.hold(False)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.show()