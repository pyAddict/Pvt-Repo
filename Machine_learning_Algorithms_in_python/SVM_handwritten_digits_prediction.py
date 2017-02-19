from sklearn import datasets
from sklearn import svm
import pandas as pd
digits=datasets.load_digits()
data=digits.data
temp=pd.DataFrame(data)
temp['Label']=digits.target
#data visualization
temp.head()
#import pylab as py
#py.gray()
#py.matshow(digits.images[1])
#py.show()
x,y=digits.data[:-1],digits.target[:-1];
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4)
sv=svm.SVC(gamma=0.0001,C=100,probability=True)
sv.fit(x_train,y_train)
y_pred=sv.predict(x_test);
print('Predicted:',sv.predict(x_test[400]))
print('Original value:',(y_test[400]))
from sklearn.metrics import accuracy_score
print("accuracy is:",accuracy_score(y_pred,y_test))
y_pred_prob=sv.predict_proba(x_test)
import pylab as py
#py.imshow(digits.images[4])
py.hist(y_pred_prob,bins=8)
py.show()
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
