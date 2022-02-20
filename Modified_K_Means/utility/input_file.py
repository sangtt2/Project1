class Data():
    def __init__(self):
        self.eg = []

def input_func(name, dataset):
    with open(name, 'r') as f:
        number = int(f.readline())
        for _ in range(number):
            l = f.readline().rstrip().split()[1:4]
            dataset.eg.append(list(map(float, l)))

