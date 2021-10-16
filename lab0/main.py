import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from knn import Knn


def load_mnist(root='./', n_samples=6e4):

    # TODO Load the MNIST dataset

    files = (('train', 'train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 6e4),
             ('test', 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 1e4))

    ret = []
    for set, imgs, labels, n in files:
        X = np.fromfile(open(root + imgs), np.uint8)
        X = X[16:].reshape((n, 28, 28)).astype(np.int)
        y = np.fromfile(open(root + labels), np.uint8)
        y = y[8:].reshape((n)).astype(np.int)
        ret.append(X[n_samples] if set == 'train' else X)
        ret.append(y[n_samples] if set == 'train' else y)
    return ret

    # End of todo


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    knn = Knn()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig = plt.subplots(nrows=4, ncols=5, sharex='all',
                       sharey='all')[1].flatten()
    for i in range(20):
        img = X_test[i]
        fig[i].set_title(y_pred[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
