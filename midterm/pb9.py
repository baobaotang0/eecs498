from matplotlib import pyplot as plt
import numpy as np
A = [np.sqrt(3), 1]
B = [1, np.sqrt(3)]
C = [np.sqrt(2), np.sqrt(2)]

x = np.array([0.1*i for i in range(20)])
b = -np.sqrt(2) -0.5 - np.sqrt(3)/2
y = - b - x

plt.plot(x, y)
plt.plot(A[0], A[1], "*")
plt.plot(B[0], B[1], "*")
plt.plot(C[0], C[1], "*")
plt.axis("equal")
plt.show()