# 使用传统的聚类方法对数据特征进行直接处理
from dataCenter import data_generator
data = data_generator()
from utils import print_labels

data = data.T
print("----data size----")
print(data.shape)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0, init='random', n_init=10)
labels = kmeans.fit_predict(data)
from utils import print_labels
print("----KMeans聚类的结果为----")
print_labels(labels)

from sklearn.cluster import AgglomerativeClustering

# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=3)
labels_hierarchical = hierarchical.fit_predict(data)
print("----层次聚类的结果为----")
print_labels(labels_hierarchical)

from sklearn.cluster import DBSCAN

# DBSCAN 聚类
dbscan = DBSCAN(eps=0.3, min_samples=1)
labels_dbscan = dbscan.fit_predict(data)
print("----DBSCAN聚类的结果为----")
print_labels(labels_dbscan)

from sklearn.mixture import GaussianMixture

# GMM 聚类
gmm = GaussianMixture(n_components=3, random_state=0)
labels_gmm = gmm.fit_predict(data)
print("----GMM聚类的结果为----")
print_labels(labels_gmm)


from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters=3, random_state=0, affinity='nearest_neighbors')
labels_spectral = spectral.fit_predict(data)
print("----谱聚类的结果为----")
print_labels(labels_spectral)

from sklearn.cluster import MeanShift
# 均值漂移聚类
mean_shift = MeanShift()
labels_mean_shift = mean_shift.fit_predict(data)
print("----均值漂移聚类的结果为----")
print_labels(labels_spectral)


from sklearn.cluster import OPTICS
# OPTICS 聚类
optics = OPTICS(min_samples=2)
labels_optics = optics.fit_predict(data)
print("----OPTICS聚类的结果为----")
print_labels(labels_spectral)

from sklearn.cluster import Birch
# Birch 聚类
birch = Birch(n_clusters=3)
labels_birch = birch.fit_predict(data)
print("----Birch聚类的结果为----")
print_labels(labels_birch)

from sklearn.cluster import AffinityPropagation
# Affinity Propagation 聚类
affinity_propagation = AffinityPropagation(random_state=0)
labels_affinity_propagation = affinity_propagation.fit_predict(data)
print("----affinity聚类的结果为----")
print_labels(labels_affinity_propagation)

from sklearn.cluster import AgglomerativeClustering
# Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=3)
labels_agglomerative = agglomerative.fit_predict(data)
print("----agglomerative聚类的结果为----")
print_labels(labels_agglomerative)

