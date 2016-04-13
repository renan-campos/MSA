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

import random

theano.config.gcc.cxxflags += " -O3 -ffast-math -ftree-loop-distribution -funroll-loops -ftracer"
theano.config.allow_gc=False
theano.config.floatX='float32'

# set OMP_NUM_THREADS to 10+ for more threads to speed up computations
theano.config.openmp = True

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

    # tried to reduce range of init values. didn't work...
    R = None

    for i in range(vocab_size):

        v = numpy.random.normal(size=(vect_size+1,1), loc=0.0, scale=.01)
        v /= numpy.linalg.norm(v)

        if R is None:
            R = v
        else:
            R = numpy.concatenate((R,v),axis=1)

    # create theta vector for each doc
    thetas = None
    ones  = numpy.ones((1,doc_count))

    for i in range(doc_count):

        dk = numpy.random.normal(size=(vect_size,1), loc=0.0, scale=.01)
        dk /= numpy.linalg.norm(dk)

        if thetas is None:
            thetas = dk
        else:
            thetas = numpy.concatenate((thetas,dk),axis=1)

    thetas = numpy.concatenate((ones,thetas))

    return thetas,R

def phi(R,w):
    """
        selects column of R representing word vector of w by computing Rw.

          w is a one-hot vector.
    """

    return numpy.dot(R,w)

def get_gradient_funcs():
    """
        finds the gradient of the cost function with respect to parameter specified in wrt
    """

    # important:

        # TODO: regularization
        # TODO: compute sentiment vectors

    # less important:

        # TODO: leverage memmap in numpy to reduce ram usage.

    _theta_reg_weight     = T.scalar("theta_reg_weight")
    _frobenius_reg_weight = T.scalar("frobenius_reg_weight")

    theta     = T.fmatrix("theta")
    _R        = T.fmatrix("_R")
    frequency = T.fmatrix("frequency")

    # obtain energies of word per document
    # row represents current document.
    # col represents word
    # we premptively remove the negative sign (-) because it cancels out.
    E_w     = T.dot(theta.T, _R)
    E_w_exp = T.exp(E_w)

    # sum the columns of above matrix together
    # obtains a vector which is the denominator of the softmax function for each doc
    E_total = T.sum(E_w_exp, 1)

    # compute probabilities for each word in their repsetive document
    probability = E_w_exp.T / E_total

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
    grad_wrt_R = theano.gradient.jacobian(cost, _R)

    grad_wrt_theta = T.grad(cost, theta)

    dcostdR     = theano.function([_R, theta, frequency, _theta_reg_weight, _frobenius_reg_weight], [cost, grad_wrt_R])
    dcostdtheta = theano.function([_R, theta, frequency, _theta_reg_weight, _frobenius_reg_weight], [cost, grad_wrt_theta])

    return dcostdR, dcostdtheta


def update_parameters(thetas,freq,theta_partitions, freq_partitions, partitioned_inds):
    """
    TODO: implement an update that only updates partitioned_columns
    """
    pass


def partition_data(thetas, freq, partition_size=200):

    num_docs = freq.shape[1]

    # indices of columns to select.
    column_indx = range(0,num_docs)

    start = 0
    end   = partition_size

    # need these for performing updates
    partitioned_inds = []

    # partitioned input.
    theta_partitions = []
    freq_partitions  = []

    while True:

        # inplace shuffling of col indices to select
        random.shuffle(column_indx)

        # get partition of indices
        partition = column_indx[start:end]

        # nothing in partition
        if len(partition) == 0:
            break

        theta_partition = None
        freq_partition  = None

        # select appropriate columns from theta and freq documents and place into partition
        for i in partition:

            c = theta[:,i]
            c = c.reshape(c.shape[0],1)

            # partition theta columns
            if theta_partition is None:
                theta_partition = c
            else:
                theta_partition = numpy.concatenate((theta_partition,c), axis=1)

            c = freq[:,i]
            c = c.reshape(c.shape[0],1)

            # partition freq columns
            if freq_partition is None:
                freq_partition = c
            else:
                freq_partition = numpy.concatenate((freq_partition,c), axis=1)

        partitioned_inds.append(partition)
        theta_partitions.append(theta_partition)
        freq_partitions.append(freq_partition)

        start = end
        end += partition_size


    return theta_partitions, freq_partitions, partitioned_inds


def gradient_ascent(R, thetas, freq, iterations=100, learning_rate=.0001, theta_reg_weight=.001, frobenius_reg_weight=.001):


    # i assume that each freq col has a corresponding theta col
    # partition_data(thetas, freq)


    # TODO: works for small data set ~1000 files. needs to be changed to stochastic gradient descent to handle
    # larger datasets (25000?)

    R_iterations = 100
    converges = lambda x1, x2: abs(x1 - x2) <= .01

    # theano functions to compute gradients
    get_dcostdR, get_dcostdtheta = get_gradient_funcs()

    for j in range(iterations):

        print "iteration: {}".format(j)

        # needs more time to converge. so there is a limit to iterations. the idea is to
        # get to correct the thetas quicker to get R converging.
        old_cost = None
        for i in range(R_iterations):
            cost, grad_wrt_R = get_dcostdR(R, thetas, freq, theta_reg_weight, frobenius_reg_weight)

            print cost

            R += learning_rate * grad_wrt_R

            if old_cost is None:
                old_cost = cost
            elif converges(old_cost, cost):
                break
            else:
                print "change in cost wrt R: ", abs(old_cost - cost)
                old_cost = cost


        # converges a lot faster. so just wait until it reaches a cost change of zero.
        old_cost = None
        while True:
            cost, grad_wrt_theta = get_dcostdtheta(R, thetas, freq, theta_reg_weight, frobenius_reg_weight)

            # don't want to update the first row of the theta matrix
            # TODO: move into theano function?
            mask = numpy.concatenate((numpy.zeros((1,grad_wrt_theta.shape[1])),
                                      numpy.ones((grad_wrt_theta.shape[0]-1,grad_wrt_theta.shape[1]))))
            thetas += learning_rate * (grad_wrt_theta * mask)

            if old_cost is None:
                old_cost = cost
            elif converges(old_cost, cost):
                break
            else:
                print "change in cost wrt theta: ", abs(old_cost - cost)
                old_cost = cost

if __name__ == "__main__":

    partition_data()

    # large example. should work!
#    freq = numpy.random.randint(low=0,high=10,size=(5000,1000))
#    theta,R = create_parameters(50,5000,1000)

#    freq = numpy.random.randint(low=0,high=10,size=(5000,25000))
#    theta,R = create_parameters(50,5000,25000)

    #freq = numpy.random.randint(low=0,high=10,size=(5,5))
    #theta,R = create_parameters(5,5,5)

#    gradient_ascent(R.astype('float32'), theta.astype('float32'), freq.astype('float32'))
    # out = gradient(R.astype('float32'), theta.astype('float32'), freq.astype('float32'), "theta")


    pass


