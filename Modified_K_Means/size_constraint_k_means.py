import numpy as np
import ast
import sys
from utility.display import display_func
from scipy.spatial.distance import cdist
from default_k_means import generate_centers
from utility.input_file import input_func, Data
from utility.calc_size import cal_size_of_cluster
from utility.calc_diameter import distance, calc_diameter_of_dataset

def init_centers(X, k):
    # select centers by default k means algorithm
    return generate_centers(X, k)

def find_labels(X, centers, max_size):
    K = centers.shape[0]
    L = len(X)
    # if L > K * max_size:
    #     messange = "Value of max size is more than " + str(L//(K))
    #     raise ValueError(messange)
    D = np.argsort(cdist(X, centers), axis=1).tolist()
    curr_size = [0] * K
    labels = []
    for i in range(len(D)):
        index = 0
        sorted_cluster = np.argsort(cdist(X[i].reshape(1,-1), centers), axis=1).reshape(-1,).tolist()
        while curr_size[sorted_cluster[index]] >= max_size:
            index += 1
        curr_size[sorted_cluster[index]] += 1
        labels.append(sorted_cluster[index])
    return np.array(labels)


def find_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) ==
        set([tuple(a) for a in new_centers]))

def size_constraint_kmeans(X, K, max_size):
    centers = init_centers(X, K)
    while True:
        labels = find_labels(X, centers, max_size)
        pre_center = centers
        centers = find_centers(X, labels, K)
        if has_converged(pre_center, centers):
            break
    return centers, labels

def sum_distances(centers, labels, dataset):
    dis = 0.0
    for i in range(len(labels)):
        add = distance(dataset[i], centers[labels[i]])
        dis += add
    return dis

def calc_diameter_of_cluster(cluster):
    res = 0.0
    l = len(cluster)
    for i in range(l):
        for j in range(l):
            res = max(res, distance(cluster[i], cluster[j]))
    return res



def main():
    args = ast.literal_eval(str(sys.argv))
    dataset = Data()
    input_func(args[1], dataset)
    X = np.array(dataset.eg)
    number_dataset = len(dataset.eg)
    alg1 = args[2]
    #alg2 = args[3]
    size = int(alg1)
    K = number_dataset // size + 1
    centers, labels = size_constraint_kmeans(X, K, size)
    print("Number of dataset: ")
    print(number_dataset)
    print("Number of clusters: ")
    print(K)
    print("Max size constraint: ")
    print(size)
    print("Min diameter, Average diameter, Max diameter: ")
    print(calc_diameter_of_dataset(dataset.eg, K, labels))
    print("Min size, Average size, Max size: ")
    print(cal_size_of_cluster(K, labels))
    print("Sum of all distances: ")
    print(sum_distances(centers, labels, dataset.eg))

if __name__ == '__main__':
    main()
