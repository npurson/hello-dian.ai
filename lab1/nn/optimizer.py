class Optim(object):

    def __init__(self):
        self.delta = None

    def step(self, delta):
        return self.delta if self.delta is not None else delta


class SGD(Optim):

    def __init__(self, momentum=0):
        super(SGD, self).__init__()
        self.momentum = momentum

    def step(self, delta):
        self.delta = (self.delta * self.momentum + delta) if self.delta is not None else delta
        return self.delta


class Adam(Optim):

    def __init__(self, momentum=0):
        super(Adam, self).__init__()
        ...

    def step(self, delta):
        ...
