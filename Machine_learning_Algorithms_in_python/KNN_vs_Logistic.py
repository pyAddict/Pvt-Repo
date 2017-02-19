from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
import pandas as pd
data_x=pd.DataFrame(x)
data_y=pd.DataFrame(y)
data_x['Y']=data_y
data_x.columns=['f1','f2','f3','f4','Y']
#Data Visualization
data_x.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
logis=LogisticRegression()
knn=KNeighborsClassifier(n_neighbors=20)
from sklearn.cross_validation import cross_val_score
scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')
print('KNN accuracy',scores.mean()*100,'%')
scores=cross_val_score(logis,x,y,cv=10,scoring='accuracy')
print('LogisticRegression accuracy',scores.mean()*100,'%')
accu=[];
for k in range (1,31):
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')
    accu.append(scores.mean())
k_range=range(1,31)
#Plotting
import matplotlib.pyplot as plt
fig = plt.figure()
%pylab qt
#from pylab import*
#plot(k_range,accu)
plt.plot(k_range,accu)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()