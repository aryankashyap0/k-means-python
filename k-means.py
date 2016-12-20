import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
from sklearn.cluster import KMeans

style.use('ggplot')

# Some X and y Values to plot on the graph
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x,y)

# for the first time running dont comment this out for the second time comment it out
plt.show()

X = np.array([[1, 2], 
	          [5, 8], 
	          [1.5, 1.8], 
	          [8, 8],
	          [1, 0.6],
	          [9, 11], 
	          ])

# k - means algorthm
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

centeroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centeroids)
print(labels)

color = ["g.", "r.", "c.","y."]

for i in range(len(X)):
	print("coordinate: ", X[i], "label:", labels[i])
	plt.plot(X[i][0], X[i][1], color[labels[i]], markersize = 10)

plt.scatter(centeroids[:, 0], centeroids[:, 1], marker= 'x', s=150, linewidths=5, zorder=10)

plt.show()