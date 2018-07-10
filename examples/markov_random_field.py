import numpy as np
import matplotlib.pyplot as plt

from archmm.mrf import MarkovRandomField


if __name__ == '__main__':

    clique = np.ones((5, 5))

    offset = 1.5
    X_train = [np.random.normal(0, 1, size=(150, 150))]
    X_train[0][50:100, 50:100] += offset

    Y_train = [np.zeros((150, 150))]
    Y_train[0][50:100, 50:100] = 1

    X_test = [np.random.normal(0, 1, size=(150, 150))]
    X_test[0][20:50, 40:110] += offset
    X_test[0][100:130, 40:110] += offset

    mrf = MarkovRandomField(
        clique=clique, max_n_iter=250, beta=0.9, t0=100., dq=0.95)
    mrf.fit(X_train, Y_train)

    # Get doubleton potentials
    # (To get the predictions, you must provide no optional argument)
    predictions = mrf.predict(X_test, rettype='dp')

    plt.subplot(1, 2, 1)
    plt.imshow(X_test[0], cmap='gray')
    plt.title("Original image")

    plt.subplot(1, 2, 2)
    plt.imshow(predictions[0][:, :, 0])
    plt.title("Doubleton potential")

    plt.show()