from sklearn import linear_model
import numpy as np
import pandas as pd
data=np.genfromtxt('exp1d2.txt',delimiter=',')
temp=pd.DataFrame(data)
#data visualization
temp.head()
x=data[:,0:2]
y=data[:,np.newaxis,2]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
lm=linear_model.LassoLars(alpha=0.1)
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
print('Variance score: %.2f' % lm.score(x_test, y_test))
#Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ax1.scatter(x_test[:,0],x_test[:,1],y_test,color='black')
ax1.plot(x_test[:,0],x_test[:,1],y_pred,color='blue')
#plt.plot(x_test,y_pred,color='blue')
plt.show()