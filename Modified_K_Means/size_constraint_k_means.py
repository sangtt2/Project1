import numpy as np
import ast
import sys
from utility.display import display_func
from scipy.spatial.distance import cdist
from default_k_means import generate_centers
from utility.input_file import input_func, Data

def init_centers(X, k):
    # select centers by default k means algorithm
    return generate_centers(X, k)

def find_labels(X, centers, max_size):
    K = centers.shape[0]
    L = len(X)
    if L > K * max_size:
        messange = "Value of max size is more than " + str(L//(K))
        raise ValueError(messange)
    D = np.argsort(cdist(X, centers), axis=1).tolist()
    curr_size = [0] * K
    labels = []
    for i in range(len(D)):
        index = 0
        while curr_size[D[i][index]] >= max_size:
            index += 1
        curr_size[D[i][index]] += 1
        labels.append(D[i][index])
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

def distance(point1, point2):
    dis = 0.0
    # eliminate the class attribute
    for i in range(len(point1)):
        add = (point1[i] - point2[i]) ** 2
        dis += add
    return dis ** 0.5

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
def calc_diameter_of_dataset(dataset, k, labels):
    res = 0.0
    x = []
    l = len(labels)
    for _ in range(k):
        x.append([])
    for i in range(l):
        x[labels[i]].append(dataset[i])
    for i in range(k):
        tmp = calc_diameter_of_cluster(x[i])
        res = max(tmp, res)
    return res
def main():
    args = ast.literal_eval(str(sys.argv))
    dataset = Data()
    input_func(args[1], dataset)
    alg1 = args[2]
    alg2 = args[3]
    K = int(alg1)
    size = int(alg2)
    X = np.array(dataset.eg)
    centers, labels = size_constraint_kmeans(X, K, size)
    print("Number of dataset: ")
    print(len(dataset.eg))
    print("Number of clusters: ")
    print(K)
    print("Diameter: ")
    print(calc_diameter_of_dataset(dataset.eg, K, labels))
    print("Max size: ")
    print(size)
    print("Sum of all distances: ")
    print(sum_distances(centers, labels, dataset.eg))


    # display_func(X, labels)
if __name__ == '__main__':
    main()
