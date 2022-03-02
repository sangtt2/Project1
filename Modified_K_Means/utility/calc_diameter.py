# find distance between 2 points
def distance(point1, point2):
    dis = 0.0
    # eliminate the class attribute
    for i in range(len(point1)):
        add = (point1[i] - point2[i]) ** 2
        dis += add
    return dis ** 0.5

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
    diameter_each_cluster = []
    l = len(labels)
    for _ in range(k):
        x.append([])
    for i in range(l):
        x[labels[i]].append(dataset[i])
    for i in range(k):
        diameter_each_cluster.append(calc_diameter_of_cluster(x[i]))
    min_diameter = min(diameter_each_cluster)
    average_diameter = sum(diameter_each_cluster)/k
    max_diameter = max(diameter_each_cluster)
    return (min_diameter, average_diameter, max_diameter)
