""" 
Keep kernel implementations in here.
"""

import numpy as np
from collections import defaultdict, Counter
from functools import wraps
from tqdm import tqdm
from scipy.sparse import csr_matrix # Added
from math import log

def cache_decorator():
    """
    Cache decorator. Stores elements to avoid repeated computations.
    For more details see: https://stackoverflow.com/questions/36684319/decorator-for-a-class-method-that-caches-return-value-after-first-access
    """
    def wrapper(function):
        """
        Return element if in cache. Otherwise compute and store.
        """
        cache = {}

        @wraps(function)
        def element(*args):
            if args in cache:
                result = cache[args]
            else:
                result = function(*args)
                cache[args] = result
            return result

        def clear():
            """
            Clear cache.
            """
            cache.clear()

        # Clear the cache
        element.clear = clear
        return element
    return wrapper


class Kernel(object):
    """ Abstract kernel object.
    """
    def evaluate(self, s, t):
        """
        Kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        raise NotImplementedError()

    def compute_kernel_matrix(self, *, X, X_prime=None):
        """
        Compute kernel matrix. Index into kernel matrix to evaluate kernel function.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Returns:
            A compressed sparse row matrix of floats with each element representing
            one kernel function evaluation.
        """
        X_prime = X if not X_prime else X_prime
        kernel_matrix = np.zeros((len(X), len(X_prime)), dtype=np.float32)

        # TODO: Implement this!

        # If X == X_prime (as it does prior to training), cheat by only computing upper-triangle of kernel matrix
        for i in range(len(X)):
          Xi = X[i]
          for j in range(len(X_prime)):
            kernel_matrix[i,j] = self.evaluate(Xi,X_prime[j])

        return csr_matrix(kernel_matrix)   
        # raise Exception("You must implement this method!")


class NgramKernel(Kernel):
    def __init__(self, *, ngram_length):
        """
        Args:
            ngram_length: length to use for n-grams
        """
        self.ngram_length = ngram_length


    def generate_ngrams(self, doc):
        """
        Generate the n-grams for a document.

        Args:
            doc: A string corresponding to a document.

        Returns:
            Set of all distinct n-grams within the document.
        """
        # TODO: Implement this!
        ngrams = set()
        for i in range(len(doc) - self.ngram_length + 1):
           ngram = ""
           for j in range(self.ngram_length):
              ngram += doc[i+j]
           ngrams.add(ngram)
        return ngrams
        # raise Exception("You must implement this method!")


    @cache_decorator()
    def evaluate(self, s, t):
        """
        n-gram kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        # TODO: Implement this!
        x = self.generate_ngrams(s)
        x_prime = self.generate_ngrams(t)
        union = x.union(x_prime)
        if len(union) == 0:
           return 1
        else:
           return len(x.intersection(x_prime)) / len(union)

        # raise Exception("You must implement this method!")


class TFIDFKernel(Kernel):
    def __init__(self, *, X, X_prime=None):
        """
        Pre-compute tf-idf values for each (document, word) pair in dataset.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Sets:
            tfidf: You will use this in the evaluate function.
        """
        self.tfidf = self.compute_tfidf(X, X_prime)
        

    def compute_tf(self, doc):
        """
        Compute the tf for each word in a particular document.
        You may choose to use or not use this helper function.

        Args:
            doc: A string corresponding to a document.

        Returns:
            A data structure containing tf values.
        """
        # TODO: Implement this!
        words_in_doc = doc.split() 
        c = Counter(words_in_doc) # How many times each word appears in the document
        tf = {}
        num_words = len(words_in_doc) # How many total words in the document
        for word in c:
           tf[word] = c[word] / num_words
        return tf
        #raise Exception("You must implement this method!")


    def compute_df(self, X, vocab):
        """
        Compute the df for each word in the vocab.
        You may choose to use or not use this helper function.

        Args:
            X: A list of strings, where each string corresponds to a document.
            vocab: A set of distinct words that occur in the corpus.

        Returns:
            A data structure containing df values.
        """
        # TODO: Implement this!
        df = {}
        for word in vocab: # Compute how many documents each word appears in 
           df[word] = 0
           for doc in X:
              doc_words = doc.split()
              if word in doc_words:
                 df[word] += 1
        return df 
        #raise Exception("You must implement this method!")


    def compute_tfidf(self, X, X_prime):
        """
        Compute the tf-idf for each (document, word) pair in dataset.
        You will call the helper functions to compute term-frequency 
        and document-frequency here.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Returns:
            A data structure containing tf-idf values. You can represent this however you like.
            If you're having trouble, you may want to consider a dictionary keyed by 
            the tuple (document, word).
        """
        # Concatenate collections of documents during testing
        if X_prime:
            X = X + X_prime

        # TODO: Implement this!
        tfidf = {}
        N = len(X)
        for doc in tqdm(X):
           tf = self.compute_tf(doc)
           vocab = set() 
           for word in tf.keys():
              vocab.add(word) # Create set of unique words in doc
           df = self.compute_df(X, vocab)
           for word, freq in tf.items():
              tfidf[(doc, word)] = freq * log(N / (df[word] + 1), 10)
        return tfidf
        #raise Exception("You must implement this method!") 


    @cache_decorator()
    def evaluate(self, s, t):
        """
        tf-idf kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        # TODO: Implement this!

        # Obtain term frequencies for each document
        tf_s = self.compute_tf(s)  
        tf_t = self.compute_tf(t)
        
        # Procedure
        k = 0
        for word in tf_s.keys():
           if word in tf_t.keys():
            freq = tf_t[word]
            k += freq * self.tfidf[(s, word)]
        return k
        #raise Exception("You must implement this method!")
