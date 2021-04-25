#IMPORTING LIBRARIES
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from google.colab import files


def convolve2D(image, filter): #DEFINING FUNCTIONS
  fX, fY = filter.shape # Get filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2) 
  newImage = np.zeros((nn,nn))
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage


train = pd.read_csv("train.csv") #CSV FILE IMPORTED
X = train.drop('label',axis=1)
Y = train['label']

#CREATE 9X9 2D ARRAY
filter = np.array([
          [1,2,2,1,3,4,3,1,1],
          [1,3,4,1,3,2,1,2,1],
          [1,1,1,1,1,1,1,1,1],
          [1,2,2,1,3,4,3,1,1],
          [1,3,4,1,3,2,1,2,1],
    [1,2,2,1,3,4,3,1,1],
    [1,3,2,1,2,3,4,2,1],
    [1,3,4,1,3,2,1,2,1],
    [1,2,2,1,3,4,3,1,1]])


X = X.to_numpy()
print(X.shape)


sX = np.empty((0,400), int)

# img = X[6]
ss = 42000 


for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))

  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,400))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]
# print(sY)
print(sY.shape)
print(sX.shape)


# train and test model
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
clf = MultinomialNB()
clf.fit(sXTrain, yTrain)
print(clf.class_count_)

print('Final Score MNB9x9 : {clf.score(sXTest, yTest)}')

print(sX.shape)

sampleSubmission=pd.read_csv('sample_submission.csv')
sampleSubmission['Label'] = p
sampleSubmission.to_csv('sub.csv', index=False)
