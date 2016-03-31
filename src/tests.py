"""
    TODO: add some tests in this document to ensure correctness of operations
          for computing the gradient
"""

import theano

A = theano.tensor.dmatrix('A')

B = theano.tensor.sum(A,1)

f = theano.function([A],B)

print f([[0,1],[0,5]])


