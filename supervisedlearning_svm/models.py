""" 
Keep model implementations in here.
"""

import numpy as np

class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, lmbda):
        self.lmbda = lmbda

    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        raise NotImplementedError()

    def predict(self, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class KernelPegasos(Model):

    def __init__(self, *, nexamples, lmbda):
        """
        Args:
            nexamples: size of example space
            lmbda: regularizer term (lambda)

        Sets:
            b: beta vector (related to alpha in dual formulation)
            t: current iteration
            kernel_degree: polynomial degree for kernel function
            support_vectors: array of support vectors
            labels_corresp_to_svs: training labels that correspond with support vectors
        """
        super().__init__(lmbda=lmbda)
        self.b = np.zeros(nexamples, dtype=int) # [ . . . . . . . . ]
        self.t = 1
        self.support_vectors = [] # samples in X with non-zero b values
        self.support_vectors_indices = [] # indices of samples in X with non-zero b values
        self.labels_corresp_to_svs = [] 


    def fit(self, *, X, y, kernel_matrix):
        """ Fit the model.

        Args:
            X: A list of strings, where each string corresponds to a document.
            y: A dense array of ints with shape [num_examples].
            kernel_matrix: an ndarray containing kernel evaluations
        """
        # TODO: Implement this!

        # Convert y's to be in {1, -1}
        y_converted = []
        for y_old in y:
           if y_old == 1: 
              y_converted.append(1)
           else: 
              y_converted.append(-1)

        # Iterate over all examples
        self.t = 1
        for j in range(len(X)):
           # Increment t and materialize sparse kernel matrix
           self.t += 1
           kernel_matrix_d = kernel_matrix.todense()
           
           # "To (update) B or not to (update) B..."
           ft = y_converted[j] / (self.lmbda*(self.t-1))
           update_condition = 0
           for i in range(len(X)):
              update_condition += self.b[i] * y_converted[i] * kernel_matrix_d[i,j]
           update_condition *= ft
           #print(update_condition)
           if update_condition < 1:
              self.b[j] += 1
              self.support_vectors.append(X[j]) # Support vectors are those data points X
                                                # with non-zero corresponding b values
              self.support_vectors_indices.append(j)
              self.labels_corresp_to_svs.append(y_converted[j])
           """
           # Increment t
           self.t += 1
           
           # "To (update) B or not to (update) B..."
           ft = y_converted[j] / (self.lmbda*(self.t-1))
           jth_col_K = kernel_matrix.todense()[:,j] # Kernel matrix is sparse
           y_dot_kj = np.dot(y_converted, jth_col_K).item()
           
           update_condition = ft * (self.b * y_dot_kj).sum()
           print("update_condition: ", update_condition, " y_converted[j]: ", y_converted[j])
           if update_condition < 1:
              self.b[j] += 1
              self.support_vectors.append(X[j]) # Support vectors are those data points X
                                                # with non-zero corresponding b values
              self.support_vectors_indices.append(j)
              self.labels_corresp_to_svs.append(y_converted[j])
           """
        # raise Exception("You must implement this method!")


    def predict(self, *, X, kernel_matrix):
        """ Predict.

        Args:
            X: A list of strings, where each string corresponds to a document.
            kernel_matrix: an ndarray containing kernel evaluations

        Returns:
            A dense array of ints with shape [num_examples].
        """
        # TODO: Implement this!

        predictions = []
        """
        print("Kernel matrix shape: " ,kernel_matrix.shape)
        print("X shape: " , len(X))
        print("Support vectors shape: " , len(self.support_vectors_indices))
        """
        for j in range(len(X)):
           tot = 0
           for i in range(len(self.support_vectors)):
              alpha = (1/(self.lmbda*self.t)) * self.b[self.support_vectors_indices[i]]
              # print(alpha)
              tot += alpha * self.labels_corresp_to_svs[i] *  kernel_matrix[i,j]
           if tot >= 0:
              predictions.append(1)
           else:
              predictions.append(0)  
        return predictions

        # raise Exception("You must implement this method!")
