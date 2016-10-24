# -*- coding: utf-8 -*-

import numpy as np


def alphaLoopedVersion(alpha, B, A):
    T, n = alpha.shape
    for t in range(1, T):
        for j in range(n):
            temp = 0
            for i in range(n):
                temp += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = B[t, j] * temp
    return alpha

def alphaVectorizedVersion(alpha, B, A):
    T, n = alpha.shape
    for t in range(1, T):
        alpha[t] = np.multiply(B[t, :], np.sum(A.T * alpha[t - 1, :], axis = 1))
    return alpha

def betaLoopedVersion(beta, B, A):
    T, n = beta.shape
    for t in range(T - 2, -1, -1):
        for i in range(n):
            beta[t, i] = 0
            for j in range(n):
                beta[t, i] += beta[t + 1, j] * A[i, j] * B[t + 1, j]
    return beta

def betaVectorizedVersion(beta, B, A):
    T, n = beta.shape
    for t in range(T - 2, -1, -1):
        # beta[t] = np.multiply(A, np.multiply(B[t + 1, :], beta[t + 1, :]))
        beta[t] = np.sum(B[t + 1, :] * (A.T * beta[t + 1, :]), axis = 1)
    return beta

if __name__ == "__main__":
    n, T = 10, 50
    B = np.random.rand(T, n)
    alpha = np.random.rand(T, n)
    beta = np.random.rand(T, n)
    A = np.random.rand(n, n)
    R1 = alphaLoopedVersion(np.copy(alpha), np.copy(B), np.copy(A))
    R2 = alphaVectorizedVersion(np.copy(alpha), np.copy(B), np.copy(A))
    L1 = betaLoopedVersion(np.copy(alpha), np.copy(B), np.copy(A))
    L2 = betaVectorizedVersion(np.copy(alpha), np.copy(B), np.copy(A))
    print((R1 == R2).all())
    print((L1 == L2).all())
    print("Finished")