#importing libraries
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#DEFINING FUNCTIONS
def convolve2D(image, filter):
  fX, fY = filter.shape 
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2)
  newImage = np.zeros((nn,nn))
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

train = pd.read_csv("train.csv") #IMPORTING CSV.FILE
X = train.drop('label',axis=1)
Y = train['label']
#2D ARRAY OF 9X9
filter = np.array([[1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1,1,1]])

X = X.to_numpy()
print(X.shape)
sX = np.empty((0,400), int)
#PRINTING X AXIS SHAPE :


ss = 500 
#FOR LOOP STARTING
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,400))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

X_train, X_test, Y_train, Y_test = train_test_split(sX,sY,test_size=0.20,random_state=1)
svcClassifier=SVC(kernel='linear')
svcClassifier.fit(X_train,Y_train)
Y_pred = svcClassifier.predict(X_test)
print('Linear Accuracy =>: ',accuracy_score(Y_test,Y_pred)*100)
#PRINTING

svcClassifier=SVC(kernel='rbf')
svcClassifier.fit(X_train,Y_train)
Y_pred = svcClassifier.predict(X_test)
print('RBF Accuracy: => ',accuracy_score(Y_test,Y_pred)*100)
#PRINTING


parameters={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}
grid=GridSearchCV(SVC(),parameters,verbose=0)
grid.fit(X_train,Y_train)
print('Best Parameters: ',grid.best_params_)
gridPred = grid.predict(X_test)
#FINAL OUTPUT
