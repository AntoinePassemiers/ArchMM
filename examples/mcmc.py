import numpy as np
import scipy.stats

from archmm.stats import MCMC


if __name__ == '__main__':

    pdf = scipy.stats.norm(loc=3., scale=0.8).pdf
    proposal = scipy.stats.norm()

    start = 0
    mcmc = MCMC(pdf, start, proposal=proposal)

    samples = mcmc.sample(200)

    fig, ax = plt.subplots(1, 1)