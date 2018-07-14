import numpy as np

from archmm.iohmm import IOHMM


if __name__ == '__main__':

    X1 = np.random.normal(0, 4, size=(150, 15))
    X2 = np.random.normal(-5, 3, size=(100, 15))
    X3 = np.empty((200, 15))
    X3[:100] = np.random.normal(0, 4, size=(100, 15))
    X3[100:] = np.random.normal(3, 2, size=(100, 15))

    Y1 = np.full(150, 0)
    Y2 = np.full(100, 1)
    Y3 = np.full(200, 0)
    Y3[100:] = 2

    X = [X1, X2, X3]
    Y = [Y1, Y2, Y3]

    iohmm = IOHMM(3)
    iohmm.fit(X, Y)
