from sklearn import datasets,linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing datasets
dibaties=datasets.load_diabetes()
#for visualising the data
data=pd.DataFrame(dibaties.data)
#data.shape
#data.head()
#choose a single feature
x=dibaties.data[:,np.newaxis,3]
# np.newaxis makes(442,) to (442,1)
#x.shape
y=dibaties.target[:,np.newaxis]
#y.shape
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
##for ordiniary regression
lm=linear_model.LinearRegression()
##for Ridge regression
#lm=linear_model.Ridge(alpha=0.5)
##for Ridge regression with CV
#lm=linear_model.RidgeCV(alphas=[1,.1,10])
#alpha=lm.alpha_
#lm=linear_model.Lasso(alpha=0.5)
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
#print('Coefficients: \n', lm.coef_)
print('Variance score: %.2f' % lm.score(x_test, y_test))
#plotting
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue',linewidth=3)
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.title('Linear regression with one feature')
plt.show()
