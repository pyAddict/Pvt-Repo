import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
data_nonLinear=np.genfromtxt('exp2d2.txt',delimiter=',')
temp=pd.DataFrame(data_nonLinear)
#Data Visualization
temp.head()
x=data_nonLinear[:,0:2]
y=data_nonLinear[:,np.newaxis,2]
y=y.ravel()
x_train,x_test,y_train,y_test=train_test_split(x,y)
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.qda import QDA
qdd=QDA()
#ldd=LinearDiscriminantAnalysis()
qdd.fit(x_train,y_train)
y_pred=qdd.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))
#Plotting
import matplotlib.pyplot as plt
x1=x_test[:,0];x2=x_test[:,1];
for i in range(0,len(y_test)):
    c=['red' if y_test[i]==1 else 'blue']
    plt.scatter(x1[i],x2[i],color=c) 
    plt.hold(True)
plt.hold(False)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()