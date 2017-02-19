import pandas as pd
url='https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data';
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima=pd.read_csv(url, header=None, names=col_names)
#Data Visualization
pima.head()
temp=['pregnant','insulin','bmi','age']
x=pima[temp]
y=pima.label
#x.head()
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
from sklearn.linear_model import LogisticRegression
logis=LogisticRegression()
logis.fit(x_train,y_train)
y_pred=logis.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))
print(y_test.values[1:25])
print(y_pred[1:25])
## confussion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
#true positive rate
from sklearn.metrics import recall_score
print (recall_score(y_test,y_pred))
print(logis.predict_proba(x_test)[0:10, :])
y_pred_prob=logis.predict_proba(x)[:,1]
#Plotting
import matplotlib.pyplot as plt
plt.hist(y_pred_prob,bins=8)
plt.show()
from sklearn.preprocessing import binarize
y_pred_class=binarize(y_pred_prob,0.3)[0]
print(y_pred_prob[1:10])
print (y_pred_class[1:10])