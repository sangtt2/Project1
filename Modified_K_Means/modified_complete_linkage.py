import sys
import ast
import numpy as np
from utility.input_file import input_func, Data
from utility.calc_size import cal_size_of_cluster
from utility.calc_diameter import distance, calc_diameter_of_dataset
from utility.display import display_func

class Cluster():
    def __init__(self, data_number):
        self.clusters = []  # list of clusters
        self.index = []  # index of the clusters, starts from 0
        self.number = data_number  # number of clusters in the cluster set
        self.labels = [0] * data_number

    def assign_label(self):
        for i in range(len(self.index)):
            for j in self.index[i]:
                self.labels[j] = i


def merge(cluster,max_diameter, max_size):
    # initialize
    min = -1
    rm_index_1 = 1
    rm_index_2 = 0
    for i, clus in enumerate(cluster.clusters):
        for j in range(cluster.number):
            dis = find_max(clus, cluster.clusters[j])
            size = len(clus) + len(cluster.clusters[j])
            if dis <= max_diameter and size <= max_size and i != j:
                if min == -1 or dis < min:
                    min = dis
                    index_list = cluster.index[i] + cluster.index[j]
                    if i < j:
                        rm_index_1 = j
                        rm_index_2 = i
                    else:
                        rm_index_1 = i
                        rm_index_2 = j
            else:
                continue
    if min != -1:
        # delete 2 clusters in index
        cluster.index.remove(cluster.index[rm_index_1])
        cluster.index.remove(cluster.index[rm_index_2])
        # store 2 clusters in a new cluster
        new_clus = cluster.clusters[rm_index_1] + cluster.clusters[rm_index_2]
        # remove 2 clusters
        cluster.clusters.remove(cluster.clusters[rm_index_1])
        cluster.clusters.remove(cluster.clusters[rm_index_2])
        # add the new cluster
        cluster.clusters.append(new_clus)
        cluster.index.append(index_list)
        cluster.number -= 1

# find 2 furtherest points between 2 clusters
def find_max(cluster1, cluster2):
    max = 0.0
    for point1 in cluster1:
        for point2 in cluster2:
            dis = distance(point1, point2)
            if max < dis:
                max = dis
    return max



def find_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def sum_distances(centers, labels, dataset):
    dis = 0.0
    for i in range(len(labels)):
        add = distance(dataset[i], centers[labels[i]])
        dis += add
    return dis



def main():
    args = ast.literal_eval(str(sys.argv))
    dataset = Data()
    alg1 = args[1] # dataset file
    alg2 = args[2] # max_diameter
    alg3 = args[3] # max_size
    input_func(args[1], dataset)  # Reading training data
    max_diameter = int(alg2)
    max_size = int(alg3)
    l = len(dataset.eg)
    cluster = Cluster(len(dataset.eg))
    for i, eg in enumerate(dataset.eg):
        cluster.clusters.append([eg])
        cluster.index.append([i])
    pre_cluster_number = cluster.number + 1
    while cluster.number < pre_cluster_number and cluster.number > 1:
        pre_cluster_number = cluster.number
        merge(cluster, max_diameter, max_size)

    cluster.assign_label()
    X = np.array(dataset.eg)
    labels = np.array(cluster.labels)
    K = cluster.number
    centers = find_centers(X, labels, K)

    print("Number of dataset: ")
    print(l)
    print("Number of clusters: ")
    print(K)
    print("Max diameter and max size constraint: ")
    print(alg2, alg3)
    print("Min diamater, Average diameter, Max diameter: ")
    print(calc_diameter_of_dataset(dataset.eg, K, labels))
    print("Min size, Average size, Max size: ")
    print(cal_size_of_cluster(K, labels))
    print("Sum of all distance: ")
    print(sum_distances(centers, labels, X))
    #display_func(X, labels)
if __name__ == "__main__":
    main()




