import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
from sklearn.cluster import KMeans

style.use('ggplot')

# Some X and y Values to plot on the graph (some random plots)
x = [1, 5, 1.5, 8, 1, 9, 12, 14, 15, 16, 18, 21, 25, 12, 36, 34, 2.3, 3.5 , 1.56, 8.75, 9.10349304, 13.56, 67.4, 75, 76, 34, 64, 56, 39, 50, 32, 56, 79, 37, 64]
y = [2, 8, 1.8, 8, 0.6, 11, 13, 16, 13, 2.4, 3.2, 1.2, 12, 76, 67, 8, 1.45, 0.65, 3.24, 1.56, 2.34, 8.90, 9.6, 32, 45 ,65, 34 ,67, 46, 54, 72, 61, 37, 43, 78]

plt.scatter(x,y)

plt.show()

X = np.array([[1, 2], 
	          [5, 8], 
	          [1.5, 1.8], 
	          [8, 8],
	          [1, 0.6],
	          [9, 11]
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