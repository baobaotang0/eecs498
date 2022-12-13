import numpy as np
from matplotlib import pyplot as plt
import numpy as np
t = [[0.27970433235168457, 324.32933831214905, 89.70713520050049, 47.65361666679382, 32.36874437332153, 143.7927839756012],
     [0.20249652862548828, 317.18688917160034, 88.43270421028137, 117.99122309684753, 30.95918846130371, 137.27017760276794]]

method = ["PCA", "MDS", "ISO", "TSNE", "LLE", "LE"]
X = np.array([i for i in range(len(method))])
plt.bar(X + 0.00, t[0], color = 'b', width = 0.25)
plt.bar(X + 0.25, t[1], color = 'g', width = 0.25)
plt.xticks(range(len(method)), method)
plt.legend(["dim=2", "dim=3"])
plt.xlabel("i am x")
plt.ylabel("i am y")
plt.title("i am title")
plt.show()
