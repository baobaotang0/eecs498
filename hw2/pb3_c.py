from sklearn.datasets import load_boston
from sklearn import preprocessing
import numpy as np

# load the boston dataset as a bunch (dictionary-like
# container object used by sklearn)
boston = load_boston()
# access the data and targets
X = boston.data
y = boston.target
X_train = X[0:400]
y_train = y[0:400]
X_test = X[400:]
y_test = y[400:]


if __name__ == '__main__':
    from sklearn import linear_model
    # instantiate and train a Lasso model
    reg = linear_model.Lasso(alpha= 1)
    reg.fit(X_train, y_train)
    # use the trained model to predict ytest from Xtest
    y_my = reg.predict(X_test)
    MSE = np.linalg.norm(y_test - y_my) ** 2 / len(y_test)
    print(MSE)
    from matplotlib import pyplot as plt
    plt.plot(y_test)
    plt.plot(y_my)
    plt.show()
