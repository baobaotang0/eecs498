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
    alpha_set = [0.5*i for i in range(10)]
    MSE_set = [0] * len(alpha_set)
    for i in range(len(alpha_set)):
        reg = linear_model.Lasso(alpha = alpha_set[i])
        reg.fit(X_train, y_train)
        y_my = reg.predict(X_test)
        MSE_set[i] = np.linalg.norm(y_test - y_my) ** 2 / len(y_test)

    from matplotlib import pyplot as plt
    plt.plot(alpha_set,MSE_set)
    plt.show()

    num_theta = 0
    reg = linear_model.Lasso(alpha=alpha_set[MSE_set.index(min(MSE_set))])
    reg.fit(X_train, y_train)
    for i in reg.coef_:
        if i != 0.0:
            num_theta += 1
    print(num_theta)


