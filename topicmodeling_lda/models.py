"""
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
np.seterr(all='raise') # Check numerical instability issues
import random
from scipy.special import digamma
from tqdm import tqdm
import scipy.sparse as sparse

class LDA(object):
    def __init__(self, *, inference):
        self.inference = inference
        self.topic_words = None


    def fit(self, *, X, iterations, estep_iterations):
        self.inference.inference(X=X, iterations=iterations, estep_iterations=estep_iterations)


    def predict(self, *, vocab, K):
        self.topic_words = {}
        preds = []
        for i, topic_dist in enumerate(self.inference.phi):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(K+1):-1]
            self.topic_words[i] = topic_words.tolist()
            preds.append('Topic {}: {}'.format(i, ' '.join(topic_words)))
        return preds


class Inference(object):
    """ Abstract inference object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, num_topics, num_docs, num_words, alpha, beta):
        self.num_topics = num_topics
        self.num_docs = num_docs
        self.num_words = num_words
        self.alpha = alpha
        self.beta = beta
        self.theta = np.zeros((num_docs, num_topics))
        self.phi = np.zeros((num_topics, num_words))

    def inference(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        raise NotImplementedError()


class MeanFieldVariationalInference(Inference):

    def __init__(self, *, num_topics, num_docs, num_words, alpha, beta, epsilon):
        super().__init__(num_topics, num_docs, num_words, alpha, beta)
        self.pi = np.zeros((self.num_docs, self.num_words, self.num_topics))
        self.gamma = np.zeros((self.num_docs, self.num_topics))
        self.lmbda = np.zeros((self.num_topics, self.num_words))
        self.sufficient_statistics = np.zeros((self.num_topics, self.num_words)) # sufficient statistics
        self.epsilon = epsilon
        self.initialize()

    def initialize_lambda(self):
        np.random.seed(0)
        self.lmbda = np.random.gamma(100, 1/100, (self.num_topics, self.num_words))

    def initialize_gamma(self):
        np.random.seed(0)
        self.gamma = np.random.gamma(100, 1/100, (self.num_docs, self.num_topics))

    def initialize(self):
        self.initialize_gamma()
        self.initialize_lambda()
        self.Eq_phi = digamma(self.lmbda) - digamma(np.sum(self.lmbda, axis=1)).reshape((self.num_topics,1))

    def inference(self, *, X, iterations, estep_iterations):
        """
        Perform Mean Field Variational Inference using EM.
        Note: labels are not used here.

        You can use tqdm using the following:
            for t in tqdm(range(iterations))

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of EM iterations
            estep_iterations: int giving max number of E-step iterations
        """
        # TODO: Implement this!

        for t in range(iterations):

            for d in range(self.num_docs):

                for m in range(estep_iterations):

                    d_word_col = X.col[np.where(X.row==d)]
                    d_word_data = X.data[np.where(X.row==d)]

                    # i
                    Eq_theta = digamma(self.gamma[d]) - digamma(np.sum(self.gamma[d]))

                    # ii 
                    # pi.shape = (D, W, K)
                    self.pi[d][d_word_col,:] = np.exp(Eq_theta.transpose()[:, np.newaxis] + self.Eq_phi[:,d_word_col]).transpose()
                    self.pi[d][d_word_col] /= np.sum(self.pi[d][d_word_col,:])
                    # iii
                    sum_gamma_prev = self.gamma[d].sum()
                    self.gamma[d] = self.alpha + np.dot(self.pi[d][d_word_col,:].transpose(), d_word_data)

                    # early stopping condition
                    delta_gamma = abs(sum_gamma_prev - self.gamma[d].sum())
                    if delta_gamma/self.num_topics < self.epsilon:
                        break

                self.sufficient_statistics[:,d_word_col] += self.pi[d][d_word_col,:].transpose() * d_word_data
            
            self.lmbda = self.beta + self.sufficient_statistics
            self.Eq_phi = digamma(self.lmbda) - digamma(np.sum(self.lmbda, axis=1)).reshape((self.num_topics,1))

        self.theta = self.gamma / self.gamma.sum(axis=0)
        self.phi = self.lmbda / self.lmbda.sum(axis=0)

        # raise NotImplementedError()

