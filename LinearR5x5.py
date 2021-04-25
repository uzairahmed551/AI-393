import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from google.colab import files


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


 train = pd.read_csv("train.csv")
X = train.drop('label',axis=1)
Y = train['label']



filter = np.array([
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1]])
filter = np.array([
          [1,2,3,4,1],
          [1,3,2,41],
          [1,1,2,3,1],
          [1,3,2,1,1],
          [1,1,1,1,1]])



X = X.to_numpy()


sX = np.empty((0,576), int)


ss = 42000


for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))


  nImg = convolve2D(img2D,filter)

  nImg1D = np.reshape(nImg, (-1,576))

  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

print(sY.shape)
print(sX.shape)


sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)

LinearRegression = LinearRegression()

LinearRegression.fit(sXTrain, yTrain)
Y_pred = LinearRegression.predict(sXTest)
print(' Final LinearRegression: {LinearRegression.score(sXTest, yTest)}')

predictedClasses = LinearRegression.predict(sXTest)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictedClasses)+1)), "Label": predictedClasses})
submissions.to_csv("submission.csv", index = False, header = True)
files.download('submission.csv')