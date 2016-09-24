# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport *
from libc.stdio cimport * 


def stableInvSigma(sigma):
    sigma = np.nan_to_num(sigma)
    singular = True
    mcv = 0.00001
    while singular:
        try:
            inv_sigma = np.array(np.linalg.inv(sigma), dtype = np.double)
            singular = False
        except np.linalg.LinAlgError:
            sigma += np.eye(len(sigma), dtype = np.double) * mcv # TODO
            mcv *= 10
    return inv_sigma