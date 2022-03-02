def cal_size_of_cluster(k, labels):
    x = [0 for _ in range(k)]
    for i in labels:
        x[i] += 1
    min_size = min(x)
    average_size = sum(x) / k
    max_size = max(x)
    return (min_size, average_size, max_size)