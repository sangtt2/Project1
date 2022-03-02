
import matplotlib.pyplot as plt
import numpy as np

def display_func(X, label):
    K = np.amax(label) + 1
    # K = 5
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    X3 = X[label == 3, :]
    X4 = X[label == 4, :]
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)
    plt.plot(X3[:, 0], X3[:, 1], 'y*', markersize=4, alpha=.8)
    plt.plot(X4[:, 0], X4[:, 1], 'b*', markersize=4, alpha=.8)
    plt.axis('equal')
    plt.plot()
    plt.show()