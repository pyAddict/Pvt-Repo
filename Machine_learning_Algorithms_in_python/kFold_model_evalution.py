import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
#Data Visualization
data.head()
%matplotlib inline
sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',size=7,aspect=0.7,kind='reg')
features=['TV','Radio','Newspaper'];
x=data[features];
#x.head()
y=data[['Sales']]
#y.head()
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
x_train.shape
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(x_train,y_train)
y_pred=lin.predict(x_test)
print (lin.coef_)
print (lin.intercept_)
zip(features,lin.coef_)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import root_mean_absolute_error
print (mean_squared_error(y_test,y_pred))
print (mean_absolute_error(y_test,y_pred))
print (np.sqrt(mean_absolute_error(y_test,y_pred)))
#Model evaluation by K-fold CV
from sklearn.cross_validation import cross_val_score
import numpy as np
scores=cross_val_score(lin,x,y,cv=10,scoring='mean_squared_error')
# as mse is negative in this case
temp=np.sqrt(-1*scores)
print('3features accuracy',temp.mean())
# as Newspaper has very less correlation with response so i am removing it  from our model
# and then check RMSE
features=['TV','Radio'];
x=data[features];
scores=cross_val_score(lin,x,y,cv=10,scoring='mean_squared_error')
temp=np.sqrt(-1*scores);
print('2features accuracy',temp.mean())
print('As we want to minimise MSE So 2 feature model is better as it has less MSE ')
features=['TV','Radio'];
x=data[features];
y=data[['Sales']]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(x_train,y_train)
y_pred=lin.predict(x_test)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print (mean_squared_error(y_test,y_pred))
print (mean_absolute_error(y_test,y_pred))
print (np.sqrt(mean_absolute_error(y_test,y_pred)))
#from sklearn.grid_search import GridSearchCV
#grid=GridSearchCV()
#features=['TV','Radio','Newpaper']
#x=data['features'];
#grid.predict([2,3,4])
