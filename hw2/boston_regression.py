import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt


boston = load_boston()
X = boston.data
y = boston.target

#Scaled data has zero mean and unit variance
X = preprocessing.scale(X)

# split the data into test and train (which includes hold out data set)
n_train_all = 400
n_val = 100

X_train_all = X[0:n_train_all]
y_train_all = y[0:n_train_all]
X_val = X[0:n_val]
y_val = y[0:n_val]
X_train = X[n_val:n_train_all]
y_train = y[n_val:n_train_all]
X_test = X[n_train_all:]
y_test = y[n_train_all:]

print('*'*25)

### perform least squares
#augment the data by 1
A_train_all = np.concatenate((np.ones((len(X_train_all), 1)),X_train_all), axis=1)
A_test = np.concatenate((np.ones((len(X_test), 1)),X_test), axis=1)
theta = np.linalg.pinv(A_train_all) @ y_train_all
print(theta)
y_predict = A_test @ theta
mse_ls = (np.linalg.norm(y_predict - y_test)**2)/len(y_test)
print('least square, MSE = %.4f' %(mse_ls))


### perform ridge regression
A_val = A_train_all[0:n_val]
A_train = A_train_all[n_val:]

lambd_grid = np.linspace(0.01, 100, 3000)
Gamma = np.identity(len(A_train[0]))

mse_rr_train = []
for i in range(0,len(lambd_grid)):
	lambd = lambd_grid[i]
	theta = np.linalg.inv(A_train.T @ A_train + Gamma * lambd) @ A_train.T @ y_train
	y_predict = A_val.dot(theta)
	mse = (np.linalg.norm(y_predict - y_val)**2)/len(y_val)
	mse_rr_train.append(mse)

plt.plot(lambd_grid,mse_rr_train)
plt.xscale('symlog')
plt.xlabel('lambd')
plt.ylabel('MSE')
plt.title('MSE on holdout set for ridge regression')
plt.xlim(min(lambd_grid), max(lambd_grid))
plt.show()

# 
min_mse = min(mse_rr_train)
min_index = mse_rr_train.index(min_mse)
lambd_best_rr = lambd_grid[min_index]
# theta = (np.linalg.inv(A_train_all.T.dot(A_train_all) + Gamma*lambd_best_rr)).dot(A_train_all.T.dot(y_train_all))
theta = np.linalg.inv(A_train_all.T @ A_train_all + Gamma * lambd_best_rr) @ A_train_all.T @ y_train_all
y_predict = A_test @ theta
mse_rr = (np.linalg.norm(y_predict - y_test)**2)/len(y_test)

print('ridge regression, lambda = %.4f, MSE = %.4f' %(lambd_best_rr,mse_rr))  



### perform Lasso
mse_lasso_train = []
for i in range(0,len(lambd_grid)):
	lambd = lambd_grid[i]
	reg = linear_model.Lasso(alpha = lambd)
	reg.fit(X_train,y_train) 
	y_predict = reg.predict(X_val)
	mse = np.sum(np.square(y_val - y_predict))/len(y_val)
	mse_lasso_train.append(mse)


plt.plot(lambd_grid,mse_lasso_train)
plt.xscale('symlog')
plt.xlabel('lambd')
plt.ylabel('MSE')
plt.title('MSE on holdout set for Lasso')
plt.xlim(min(lambd_grid), max(lambd_grid))
plt.show()
min_mse = min(mse_lasso_train)
min_index = mse_lasso_train.index(min_mse)
lambd_best_lasso = lambd_grid[min_index]
reg = linear_model.Lasso(alpha = lambd_best_lasso)
reg.fit(X_train_all,y_train_all) 
y_predict = reg.predict(X_test)
mse_lasso = (np.linalg.norm(y_predict - y_test)**2)/len(y_test)
print('Lasso coefficient is', reg.coef_)
print('Lasso, lambda = %.4f, MSE = %.4f, number of non-zero elements in theta = %d' %(lambd_best_lasso,mse_lasso, np.count_nonzero(reg.coef_)))  
