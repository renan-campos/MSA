"""
 Text-Machine Lab: MSA

 File Name : learning.py

 Creation Date : 27-03-2016

 Created By : Kevin Wacome

 Purpose : This module contains code for learning word vectors

           as specified within: http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf

"""

# TODO: need to fix memory issues

import time
import numpy
import theano
import theano.tensor as T

B_t = None


def create_parameters(vect_size,vocab_size,doc_count):
    """
        creates a Gaussian Matrix with dimensions:
            vect_size+1: where vect_size is the size of the word vector.
            vocab_size  : is the size of the vocabulary of corpus.

        creates a theta Matrix of dimension vect_size+1 x doc_count, where vect_size is the size of our word vector and
        doc_count is the number of documents. each document will have its own theta vector.

        a row constant of 1 is added to the theta matrix account for the bias within the R matrix.
    """
    # ***************************************************************************************************
    # ************NOTE: make sure we cut the bias off when computing final result...*********************
    # ***************************************************************************************************
    R    = numpy.random.randn(vect_size,vocab_size)
    bias = numpy.random.randn(1,vocab_size)

    thetas = numpy.random.randn(vect_size,doc_count)
    ones  = numpy.ones((1,doc_count))

    thetas = numpy.concatenate((ones,thetas))
    R     = numpy.concatenate((bias, R))

    return thetas,R

def E_all(R,thetas):
    """
        generates energy of word.

        selects column of R representing word vector of w by computing Rw.
        w is a one-hot vector.

        this implementation deviates from the paper by already including the bias term
        within phi.

        then computes vector product of -(theta * phi)
    """

    R_t = numpy.transpose(R)

    return (-(numpy.dot(R_t,theta)))

def phi(R,w):
    """
        selects column of R representing word vector of w by computing Rw.

          w is a one-hot vector.
    """

    return numpy.dot(R,w)

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

def gradient_R(R, thetas):

    # TODO: better comments
    # TODO: regularization

    theta = T.dmatrix('theta')
    _R = T.dmatrix('R')

    # find f' where f = log(e^blah[x]/sum(e^blah[i], i))

    # obtain energies of word per document
    # row represents current document.
    # col represents word
    # we premptively remove the negative sign (-) because it cancels out.
    E_w = T.dot(theta.T, _R)

    # sum the columns of above matrix together
    # obtains a vector which is the denominator of the softmax function for each doc
    E_total = T.sum(T.exp(E_w))

    # TODO: why do we need this. is E_total not the same thing?
    E_tv = E_w.fill(E_total)

    # compute probabilities for each word in their repsetive document
    # TODO: should be T.exp(-E_w)? unless we remove negative sign on line 104?
    probability = T.exp(E_w) / E_tv

    # computes cost for each document
    # TODO: we need to multiply by a frequency matrix. because we don't account for how many times a word occurs in a doc
    # we just assume once for now.
    cost = T.sum(T.log(probability))

    # compute gradient of each document wrt each element in R
    grad = T.grad(cost, _R)

    dcostdR = theano.function([theta, _R], grad)

    return dcostdR(thetas, R)

if __name__ == "__main__":

    theta,R = create_parameters(50,5000,75000)

    init_time = time.time()

    print gradient_R(R, theta)

    print time.time() - init_time

    pass


