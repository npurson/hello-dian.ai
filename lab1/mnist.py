import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import nn
import nn.functional as F


n_features = 28 * 28
n_classes = 10
n_epochs = 10
bs = 1000
lr = 1e-3
lengths = (n_features, 512, n_classes)


class Model(nn.Module):

    # TODO Design the classifier.

    ...

    # End of todo


def load_mnist(mode='train', n_samples=None, flatten=True):
    images = './train-images-idx3-ubyte' if mode == 'train' else './t10k-images-idx3-ubyte'
    labels = './train-labels-idx1-ubyte' if mode == 'train' else './t10k-labels-idx1-ubyte'
    length = 60000 if mode == 'train' else 10000

    X = np.fromfile(open(images), np.uint8)[16:].reshape(
        (length, 28, 28)).astype(np.int32)
    if flatten:
        X = X.reshape(length, -1)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape(
        (length)).astype(np.int32)
    return (X[:n_samples] if n_samples is not None else X,
            y[:n_samples] if n_samples is not None else y)


def vis_demo(model):
    X, y = load_mnist('test', 20)
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    fig = plt.subplots(nrows=4, ncols=5, sharex='all',
                       sharey='all')[1].flatten()
    for i in range(20):
        img = X[i].reshape(28, 28)
        fig[i].set_title(preds[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.savefig("vis.png")
    plt.show()


def main():
    trainloader = nn.data.DataLoader(load_mnist('train'), batch=bs)
    testloader = nn.data.DataLoader(load_mnist('test'))
    model = Model(lengths)
    optimizer = nn.optim.SGD(model, lr=lr, momentum=0.9)
    criterion = F.CrossEntropyLoss(n_classes=n_classes)

    for i in range(n_epochs):
        bar = tqdm(trainloader, total=6e4 / bs)
        bar.set_description(f'epoch  {i:2}')
        for X, y in bar:
            probs = model.forward(X)
            loss = criterion(probs, y)
            model.backward(loss.backward())
            optimizer.step()
            preds = np.argmax(probs, axis=1)
            bar.set_postfix_str(f'acc={np.sum(preds == y) / len(y) * 100:.1f}'
                                ' loss={loss.value:.3f}')

        for X, y in testloader:
            probs = model.forward(X)
            preds = np.argmax(probs, axis=1)
            print(f' test acc: {np.sum(preds == y) / len(y) * 100:.1f}')

    vis_demo(model)


if __name__ == '__main__':
    main()
