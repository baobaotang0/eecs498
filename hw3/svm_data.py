import numpy as np
import sklearn
from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn import preprocessing
from scipy.io import arff
import matplotlib.pyplot as plt

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, data_home="scikit_learn_data/")
print("reading ends")

# # plot one digital image
# j = 1
# plt.title('The jth image is a {label}'.format(label=int(y[j])))
# plt.imshow(X[j].reshape((28,28)), cmap='gray')
# plt.show()

# Preprocessing: scale data with zero mean and unit variance
X = preprocessing.scale(X)

# Extract out the digits "4" and "9"
X4 = X[y == '4', :]
X9 = X[y == '9', :]
y4 = 4 * np.ones((len(X4),), dtype=int)
y9 = 9 * np.ones((len(X9),), dtype=int)

# Decide on a finite set of “grid points" on which to test C
C_grid = np.logspace(-3, 3, 10)

# # split the data into test and train (which further splitted into train and validation)
X4_train = X4[0:3000, :]
X4_val = X4[3000:4000, :]
X4_test = X4[4000:, :]
X9_train = X9[0:3000, :]
X9_val = X9[3000:4000, :]
X9_test = X9[4000:, :]

X_train = np.concatenate((X4_train, X9_train), axis=0)
y_train = np.concatenate((y4[:3000], y9[:3000]), axis=0)
X_val = np.concatenate((X4_val, X9_val), axis=0)
y_val = np.concatenate((y4[3000:4000], y9[3000:4000]), axis=0)
X_test = np.concatenate((X4_test, X9_test), axis=0)
y_test = np.concatenate((y4[4000:], y9[4000:]), axis=0)
X_train_whole = np.concatenate((X_train, X_val), axis=0)
y_train_whole = np.concatenate((y_train, y_val), axis=0)

###pb_a
print("pb2(a)")
error_list = [0] * len(C_grid)
for i in range(len(C_grid)):
    clf = svm.SVC(C=C_grid[i], kernel='poly', degree=1)
    clf.fit(X_train, y_train)
    error_list[i] = 1 - clf.score(X_val, y_val)

# Report the best value of C
plt.title('validation error vs. C')
plt.xlabel('C')
plt.ylabel('Validation error')
# plt.xlim(1e-4, 1e+3)
# plt.ylim(0, 0.15)
plt.plot(C_grid, error_list)
plt.show()

# Best value of C
Best_C = C_grid[error_list.index(min(error_list))]
print("Best value of C:", Best_C)

# Retrain the whole dataset with best C
clf = svm.SVC(C=Best_C, kernel='poly', degree=1)
clf.fit(X_train_whole, y_train_whole)

# Testing
Test_error = 1 - clf.score(X_test, y_test)
print("Test error corresponding to the best C:", Test_error)


#Best value of C: 2.154434690031882
#Test error corresponding to the best C: 0.027499135247319284