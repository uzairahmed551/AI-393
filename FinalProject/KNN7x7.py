#IMPORTING LIBRARIES
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

train = pd.read_csv("train.csv") #IMPORT TRAIN.SCV
X = train.drop('label',axis=1)
Y = train['label']


filter = np.array([[1,1,1,1,1,1,1], #2D ARRAY 7X7 KNN
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1],
          [1,1,1,1,1,1,1]])


X = X.to_numpy()
print(X.shape)

sX = np.empty((0,484), int)


ss = 28000 

for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,484))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

print(sY.shape)
print(sX.shape)
#PRINT X AND Y AXIS


X_train, X_test, Y_train, Y_test = train_test_split(sX,sY,test_size=0.1)
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
print(X_test.shape)
X_test = scaler.fit_transform(X_test)

print('Length: ',len(Y_test)) #LENGTH
print('K: ',np.sqrt(len(Y_test)))

classifier = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(classification_report(Y_test,Y_pred))
print("Accuracy: ", end= ' ')
print(accuracy_score(Y_test,Y_pred)*100)
#PRINTING ALL FINAL RESULTS
