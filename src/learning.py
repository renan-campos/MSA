"""
 Text-Machine Lab: MSA

 File Name : learning.py

 Creation Date : 27-03-2016

 Created By : Kevin Wacome

 Purpose : This module contains code for learning word vectors

           as specified within: http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf

"""

import numpy

def create_parameters(B,V,D):
    """
        creates a Gaussian Matrix with dimensions:
            B+1: where B is the size of the word vector.
            V  : is the size of the vocabulary of corpus.

        creates a theta Matrix of dimension B+1 x D, where B is the size of our word vector and
        D is the number of documents. each document will have its own theta vector.

        a row constant of 1 is added to the theta matrix account for the bias within the R matrix.
    """

    # NOTE: make sure we cut the bias off when computing final result...
    R    = numpy.random.randn(B,V)
    bias = numpy.random.randn(1,V)

    theta = numpy.random.randn(B,D)
    ones  = numpy.ones((1,D))

    theta = numpy.concatenate((ones,theta))
    R     = numpy.concatenate((bias, R))

    return theta,R

def E(w,theta,R):
    """
        generates energy of word.

        selects column of R representing word vector of w by computing Rw.
        w is a one-hot vector.

        this implementation deviates from the paper by already including the bias term
        within phi.

        then computes vector product of -(theta * phi)
    """

    phi = numpy.dot(R,w)

    return (-(numpy.dot(phi,theta)))[0]

# TODO: currently verifying my vectorized implementation by hand....
# def gradient()

if __name__ == "__main__":

    theta,R = create_parameters(5,3, 2)

#    print E([0,1,0],theta,R)
    pass

