import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

class Clustering:
    def __init__(self):
        pass
    
    def auto_cluster(self, data, method='kmeans', n_clusters=3):
        if method == 'kmeans':
            return self.kmeans(data, n_clusters)
        elif method == 'dbscan':
            return self.dbscan(data)
        elif method == 'agglomerative':
            return self.agglomerative(data, n_clusters)
        elif method == 'gmm':
            return self.gmm(data, n_clusters)
        else:
            raise ValueError('Invalid method')
    
    def kmeans(self, data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data)
        return labels, kmeans
    
    def dbscan(self, data):
        dbscan = DBSCAN()
        labels = dbscan.fit_predict(data)
        return labels, dbscan

    def agglomerative(self, data, n_clusters):
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(data)
        return labels, agglomerative
    
    def gmm(self, data, n_clusters):
        gmm = GaussianMixture(n_components=n_clusters)
        labels = gmm.fit_predict(data)
        return labels, gmm
