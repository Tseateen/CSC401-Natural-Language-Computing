from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random

from scipy.special import logsumexp

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        self.precompute = ((self.mu ** 2) / (2 * self.Sigma)).sum(axis=1) + \
                          (self._d / 2) * np.log(2 * np.pi) + np.log(self.Sigma[m]).sum() / 2
        return self.precompute[m]

    def reset_pre(self):
        self.precompute = self.precompute = self._d / 2 * np.log(2 * np.pi) + np.sum(np.power(self.mu, 2)
                                                                                     / (2 * self.Sigma)) + np.sum(
            np.log(self.Sigma)) / 2

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    return - ((0.5 * (np.einsum('ij,ij->i', x / myTheta.Sigma[m][np.newaxis, :], x))) -
              (np.einsum('ij,ij->i', myTheta.mu[m] / myTheta.Sigma[m][np.newaxis, :], x))) \
           - myTheta.precomputedForM(m)


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    return log_Bs + np.log(myTheta.omega) - np.max(log_Bs + np.log(myTheta.omega), axis=0, keepdims=True) + \
           logsumexp(log_Bs + np.log(myTheta.omega) - np.max(log_Bs + np.log(myTheta.omega), axis=0, keepdims=True),
                     axis=0, keepdims=True)


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """

    return np.sum(logsumexp(log_Bs + np.log(myTheta.omega), axis=0, keepdims=True))


def Update(mt, X, lp):
    mu_reset = np.exp(lp).dot(X) / (np.exp(lp).sum(axis=1, keepdims=True))
    mt.reset_mu(mu_reset)
    mt.reset_omega((np.sum(np.exp(lp), axis=1) / lp.shape[1]).reshape((lp.shape[0], 1)))
    mt.reset_Sigma((np.exp(lp).dot(np.power(X, 2)) / (np.exp(lp).sum(axis=1, keepdims=True))) - np.power(mt.mu, 2))
    return mt


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)
    myTheta.reset_omega(np.ones(myTheta.omega.shape) / M)
    myTheta.reset_mu(X[np.random.randint(0, X.shape[0], M)])
    myTheta.reset_Sigma(np.ones((M, X.shape[1])))

    # for ex.,
    # myTheta.reset_omega(omegas_with_constraints)
    # myTheta.reset_mu(mu_computed_using_data)
    # myTheta.reset_Sigma(some_appropriate_sigma)

    index = 0
    pl = -np.inf
    change = np.inf

    while index < maxIter and change >= epsilon:
        myTheta.reset_pre()
        logb = np.array([log_b_m_x(m, X, myTheta) for m in range(M)])
        logp = log_p_m_x(logb, myTheta)
        logb = logLik(logb, myTheta)

        old = logLik(logb, myTheta)
        myTheta = Update(myTheta, X, logp)
        change = old - pl
        pl = old
        index += 1
    myTheta.reset_pre()
    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    lst = []
    index = 0
    for model in models:
        lst.append((index, logLik(np.array([log_b_m_x(m, mfcc, model) for m in range(model._M)]), model), model))
        index += 1

    lst = sorted(lst, key=lambda tup: tup[1], reverse=True)
    bestModel = lst[0][0] if lst else -1

    print(models[correctID].name)

    for ik in range(k):
        print(f"{lst[ik][2].name} {lst[ik][1]}")

    if k > 0:
        print("")

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    # print("Speaker:{} Accuracy: {}".format(k, accuracy))
