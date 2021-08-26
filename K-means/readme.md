# K-means algorithm
------------
### Background 
K-means algoritm is unsupervised machine learning algoritm used to claster the data. The number of clusters is predefined with constant k. Steps in algoritm: 
1) Pick centroid of every cluster by randomly sampling points from the dataset(X)
    '''
    centroids = self.X[np.random.randint(0,self.X.shape[0],self.k)]
    '''
2) Calculate the euclidean distance between centroid and points(X) \sqrt{(centroidX - pointsX)^2+(centroidY - pointsY)^2}
   '''
   distances = np.array([np.power(np.sum((self.X - old_centroids[i])**2,axis = 1),.5) for i in range(self.k)])
   '''
3) Attach each point to the cluster by the smallest distance using argmax function
   '''
   clusters_idx = np.argmin(distances,axis = 0)
   clusters = [self.X[np.where(clusters_idx == i)] for i in range(self.k)]
   '''
4) Update the centroids by calculating the mean of the points in the cluster
   '''
   new_centroids = np.array([np.mean(clusters[i], axis = 0) for i in range(self.k)])
   '''
5) Repeat this process over predefined no. of iterations or till centroid will not change value over one iteration.

### Results
------
![gif](https://github.com/maciejbalawejder/MLalgorithms-collection/blob/main/K-means/mygif.gif)
