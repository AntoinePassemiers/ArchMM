import numpy as np
import matplotlib.pyplot as plt

from archmm.mrf import MarkovRandomField


if __name__ == '__main__':

    clique = np.ones((7, 7))

    X_train = [np.random.rand(150, 150)]
    X_train[0][50:100, 50:100] += 0.4

    Y_train = [np.zeros((150, 150))]
    Y_train[0][50:100, 50:100] = 1

    X_test = [np.random.rand(150, 150)]
    X_test[0][20:50, 40:110] += 0.4
    X_test[0][100:130, 40:110] += 0.4

    mrf = MarkovRandomField(
        clique=clique, max_n_iter=250, beta=1.3, t0=10., dq=0.95)
    mrf.fit(X_train, Y_train)

    predictions = mrf.predict(X_test)

    plt.subplot(1, 2, 1)
    plt.imshow(X_test[0], cmap='gray')
    plt.title("Original image")

    plt.subplot(1, 2, 2)
    plt.imshow(predictions[0][:, :])
    plt.title("Segmented image")

    plt.show()