import numpy as np
import matplotlib.pyplot as plt

from archmm.mrf import MarkovRandomField


if __name__ == '__main__':

    clique = np.ones((10, 10))

    X_train = [np.random.rand(150, 150)]
    X_train[0][50:100, 50:100] += 0.5

    Y_train = [np.zeros((150, 150))]
    Y_train[0][50:100, 50:100] = 1

    X_test = [np.random.rand(150, 150)]
    X_test[0][20:50, 40:110] += 0.5
    X_test[0][100:130, 40:110] += 0.5

    mrf = MarkovRandomField(clique=clique, max_n_iter=100, beta=0.1)
    mrf.fit(X_train, Y_train)

    proba = mrf.predict(X_test) # rettype='proba')

    plt.subplot(1, 2, 1)
    plt.imshow(X_test[0], cmap='gray')
    plt.title("Original image")

    plt.subplot(1, 2, 2)
    plt.imshow(proba[0][:, :])
    plt.title("Segmented image")

    plt.show()