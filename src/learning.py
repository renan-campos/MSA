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

def gradient(R, thetas, freq, wrt, theta_reg_weight=.01, frobenius_reg_weight=.01):
    """
        finds the gradient of the cost function with respect to parameter specified in wrt
    """

    # important:

        # TODO: regularization
        # TODO: compute sentiment vectors
        # TODO: make thetas and R shared variables so they can be recomputed for N iterations

    # less important:

        # TODO: figure out how to reduce memory consumption

    _theta_reg_weight = T.scalar("theta_reg_weight")
    _frobenius_reg_weight = T.scalar("frobenius_reg_weight")

    theta = T.dmatrix('theta')
    _R = T.dmatrix('R')
    frequency = T.dmatrix('frequency')

    # obtain energies of word per document
    # row represents current document.
    # col represents word
    # we premptively remove the negative sign (-) because it cancels out.
    E_w = T.exp(T.dot(theta.T, _R))

    # sum the columns of above matrix together
    # obtains a vector which is the denominator of the softmax function for each doc
    E_total = T.sum(E_w, 1)

    # compute probabilities for each word in their repsetive document
    probability = E_w.T / E_total

    # take the log of the probability before multiplying by frequency
    log_prob = T.log(probability)

    # multiply by frequency to account multiple occurances of the same owrd, and words that
    # do not appear
    weighted_prob = log_prob * frequency

    # the paper says that for each doc you take the theta_k
    # vector and compute the euclidean norm and square that.
    # so what we are calculating is the sum of the square of each vector element
    # since we are doing this for each document and addition is commutative
    # I've brought the regularization term out front.
    theta_reg = _theta_reg_weight * T.sum(theano.tensor.pow(theta,2))

    # the frobenius norm is just the summation of the square of all of the elements in a matrix
    # since we are squaring this norm and because addition is commutative we can just do an element
    # wise squaring and thne just add all of teh elements
    frobenius_reg = _frobenius_reg_weight * T.sum(theano.tensor.pow(_R,2))

    # computes total cost for all document
    cost = frobenius_reg + theta_reg + T.sum(weighted_prob)

    # compute gradient of each document wrt each element in the specified variable (R or theta)
    # TODO: splice out bias term when computing gradient of theta
    if wrt is "R":
        grad = T.grad(cost, _R)
    elif wrt is "theta":
        grad = T.grad(cost, theta)
    else:
        print "ERROR: Gradient cannot be computed with respect to %s" % wrt
        exit(1)

    dcostdR = theano.function([theta, _R, frequency, _theta_reg_weight, _frobenius_reg_weight], grad)

    # leave as is
    if wrt == "R":
        return dcostdR(thetas, R, freq, theta_reg_weight, frobenius_reg_weight)
    else:
        theta_grad = dcostdR(thetas, R, freq, theta_reg_weight, frobenius_reg_weight)

        # don't want to update the first row of the theta matrix
        mask = numpy.concatenate((numpy.zeros((1,theta_grad.shape[1])),
                                  numpy.ones((theta_grad.shape[0]-1,theta_grad.shape[1]))))

        return theta_grad * mask


if __name__ == "__main__":

    # large example. should work!
    freq = numpy.random.randn(5000,25000)
    theta,R = create_parameters(50,5000,25000)

    # smaller dev example
    # freq = numpy.random.randn(2,2)
    # theta,R = create_parameters (2,2,2)

    print "R.shape: ", R.shape
    print "theta.shape: ", theta.shape

    init_time = time.time()

    out = gradient(R, theta, freq, "R")

    print "Gradient of R: "
    print out
    # print out.shape
    print "Time elapsed: ", time.time() - init_time

    init_time = time.time()

    out = gradient(R, theta, freq, "theta")

    print "Gradient of theta: "
    print out
    # print out.shape
    print "Time elapsed: ", time.time() - init_time

    pass


