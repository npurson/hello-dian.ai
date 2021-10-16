class DataLoader(object):

    def __init__(self, data, batch=None):
        self.X, self.y = data
        self.b = batch

    def __iter__(self):
        if self.b is None:
            yield self.X, self.y
        else:
            n = 0
            while n + self.b <= self.X.shape[0]:
                yield self.X[n:n + self.b], self.y[n:n + self.b]
                n += self.b

    def __len__(self):
        return (self.X.shape[0] / self.b) if self.b is not None else 1
