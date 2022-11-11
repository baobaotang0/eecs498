from matplotlib import pyplot as plt

x = [[-4, -6, -3, -1, 1, -3, 0, 2, 4, 2, 5, 3], [5, -1, 2, 1, -2, -4, 3, 3, 1, -5, 4, -4]]
y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


for i in range(len(y)):
    x0 = x[0][i]
    x1 = x[1][i]
    if y[i]:
        plt.plot(x0,x1,"r*")
    else:
        plt.plot(x0, x1, "bo")
plt.axis("equal")
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()