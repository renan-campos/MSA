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

import copy

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

        # adding 2 to account for both bias terms
        v = numpy.random.normal(size=(vect_size+2,1), loc=0.0, scale=.01)
        v /= numpy.linalg.norm(v)

        if R is None:
            R = v
        else:
            R = numpy.concatenate((R,v),axis=1)

    ones   = numpy.ones((1,doc_count))
    zeros  = numpy.zeros((1,doc_count))
    # create theta vector for each doc
    thetas = None

    for i in range(doc_count):

        dk = numpy.random.normal(size=(vect_size,1), loc=0.0, scale=.01)
        dk /= numpy.linalg.norm(dk)

        if thetas is None:
            thetas = dk
        else:
            thetas = numpy.concatenate((thetas,dk),axis=1)

    # adding col of 1s and col of 0s to keep bias used for energy and throw out sentiment bias terms
    thetas = numpy.concatenate((ones,zeros,thetas))

    # create psi vector for each doc
    psis = None

    for i in range(doc_count):

        psi = numpy.random.normal(size=(vect_size,1), loc=0.0, scale=.01)
        psi /= numpy.linalg.norm(psi)

        if psis is None:
            psis = psi
        else:
            psis = numpy.concatenate((psis, psi), axis=1)

    # adding col of 0s and col of 1s to keep bias for seniment and throw out bais for energy
    psis = numpy.concatenate((zeros,ones,psis))

    return thetas,R,psis

def get_sentiment_weights(no_unsup, no_pos, no_neg):
    """
        creates column vector for weighting sentiment cost calculation by number of reviews
        of each rating. Assumes document matrix is constructed first with unspecified reviews,
        then with posative reviews, and lastly with negative reviews
    """

    # find weight for each category. unsup weight is set to 0 so that sentiment data will not effect
    # cost of reviews where no sentiment score is provided
    unsup_weight = 0.0
    pos_weight = 1.0/no_pos
    neg_weight = 1.0/no_neg

    # create vector for each weight
    weights_unsup = numpy.empty((1, no_unsup))
    weights_pos   = numpy.empty((1, no_pos))
    weights_neg   = numpy.empty((1, no_neg))

    # fill vectors with correct weights
    weights_unsup.fill(unsup_weight)
    weights_pos.fill(pos_weight)
    weights_neg.fill(neg_weight)

    # create final weight vector
    sentiment_weights = numpy.concatenate((weights_unsup, weights_pos, weights_neg), axis=1)

    return sentiment_weights

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

    theta        = T.fmatrix("theta")
    _R           = T.fmatrix("_R")
    frequency    = T.fmatrix("frequency")
    psi          = T.fmatrix("psi")
    sent_weights = T.fmatrix("sent_weights")

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
    theta_reg = _theta_reg_weight * T.sum(T.pow(theta,2))

    # the frobenius norm is just the summation of the square of all of the elements in a matrix
    # since we are squaring this norm and because addition is commutative we can just do an element
    # wise squaring and thne just add all of teh elements

    # splicing out first two rows. authors do not regularize the biases.
    frobenius_reg = _frobenius_reg_weight * T.sum(T.pow(_R[2:,:],2))

    # apply sentiment weights to each word
    sentiment = T.dot(psi.T, _R)

    # compute probabilities of each words sentiment using sigmoid
    sentiment_probability = T.nnet.sigmoid(sentiment)

    # take the log of the probability before multiplying by the frequency of each sentiment
    log_sent = T.log(sentiment_probability)

    # sum sentiment probabilities to get the probability of a given document's sentiment
    # weight by the 1/number of same rating (pos/neg) documents. This is 0 for unlabled documents
    # so that sentiment is ignored when no sentiment data is present
    doc_sent_prob = sent_weights * T.sum(log_sent, 1)

    # computes total cost for all document
    cost = frobenius_reg + theta_reg + T.sum(weighted_prob) + T.sum(doc_sent_prob)

    # compute gradient of each document wrt each element in the specified variable (R or theta or psi)
    grad_wrt_R = theano.gradient.jacobian(cost, _R)
    grad_wrt_psi = theano.gradient.jacobian(cost, psi)
    grad_wrt_theta = T.grad(cost, theta)

    dcostdR     = theano.function([_R, theta, frequency, _theta_reg_weight, _frobenius_reg_weight, psi, sent_weights], [cost,grad_wrt_R])
    dcostdtheta = theano.function([_R, theta, frequency, _theta_reg_weight, _frobenius_reg_weight, psi, sent_weights], [cost,grad_wrt_theta])
    dcostdpsi   = theano.function([_R, theta, frequency, _theta_reg_weight, _frobenius_reg_weight, psi, sent_weights], [cost,grad_wrt_psi])
    cost_func   = theano.function([_R, theta, frequency, _theta_reg_weight, _frobenius_reg_weight, psi, sent_weights], [cost])

    return dcostdR, dcostdtheta, dcostdpsi, cost_func


def sample_data(thetas, freq, sample_size=600):
    """
    Sample columns.
    """

    num_docs = freq.shape[1]

    # indices of columns to select.
    column_indx = range(0,num_docs)

    start = 0
    end   = sample_size

    # inplace shuffling of col indices to select
    random.shuffle(column_indx)

    # get sample idices
    sample_inds = column_indx[0:sample_size]

    theta_samples = None
    freq_samples  = None

    # select appropriate columns from theta and freq documents and place into partition
    for i in sample_inds:

        c = thetas[:,i]
        c = c.reshape(c.shape[0],1)

        # partition theta columns
        if theta_samples is None:
            theta_samples = c
        else:
            theta_samples = numpy.concatenate((theta_samples,c), axis=1)

        c = freq[:,i]
        c = c.reshape(c.shape[0],1)

        # partition freq columns
        if freq_samples is None:
            freq_samples = c
        else:
            freq_samples = numpy.concatenate((freq_samples,c), axis=1)

    return theta_samples.astype('float32'), freq_samples.astype('float32'), sample_inds


def partition_data(thetas, freq, psi, sent_weights, partition_size=1000):
    """
    Partition thetas and freq matrices into smaller matrices of column len partition_size.
    Indices of original position within thetas and freqs are returned aswell.
    """

    num_docs = freq.shape[1]

    # indices of columns to select.
    column_indx = range(0,num_docs)

#    print column_indx

    start = 0
    end   = partition_size

    # need these for performing updates
    partitioned_inds = []

    # partitioned input.
    theta_partitions  = []
    freq_partitions   = []
    psi_partitions    = []
    weight_partitions = []

    # inplace shuffling of col indices to select
    random.shuffle(column_indx)

    while True:

        # get partition of indices
        partition = column_indx[start:end]

        # nothing in partition
        if len(partition) == 0:
            break

        theta_partition  = None
        freq_partition   = None
        psi_partition    = None
        weight_partition = None

        # select appropriate columns from theta and freq documents and place into partition
        for i in partition:

#            print "thetas: ", thetas
#            print "thetas.shape: ", thetas.shape

#            print "i: ", i

            c = thetas[:,i]
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

            c = psi[:,i]
            c = c.reshape(c.shape[0],1)

            # partition psi columns
            if psi_partition is None:
                psi_partition = c
            else:
                psi_partition = numpy.concatenate((psi_partition,c), axis=1)

            c = sent_weights[:,i]
            c = c.reshape(c.shape[0],1)

            # partition weight columns
            if weight_partition is None:
                weight_partition = c
            else:
                weight_partition = numpy.concatenate((weight_partition,c), axis=1)

        partitioned_inds.append(partition)
        theta_partitions.append(theta_partition)
        freq_partitions.append(freq_partition)
        psi_partitions.append(psi_partition)
        weight_partitions.append(weight_partition)

        start = end
        end += partition_size

    return theta_partitions, freq_partitions, psi_partitions, weight_partitions, partitioned_inds


def update_parameter(parameter, grad_update, sample_inds):

    for i, j in zip(sample_inds,range(0,len(sample_inds))):
        parameter[:,i] = grad_update[:,j]

    return parameter

def gradient_ascent(R, thetas, freq, psi, sent_weights, iterations=100, parameter_iterations=30, partition_size=1000, learning_rate=1e-4, theta_reg_weight=.01, frobenius_reg_weight=.01):

    converges = lambda x1, x2: abs(x1 - x2) <= .01

    # theano functions to compute gradients
    get_dcostdR, get_dcostdtheta, get_dcostdpsi, cost_func = get_gradient_funcs()

    for i in range(iterations):

        # needs more time to converge. so there is a limit to iterations. the idea is to
        # get to correct the thetas quicker to get R converging.
        old_cost = None

        # sample thetas and freq because they correspond to document we are training on.
        theta_partitions, freq_partitions, psi_partitions, sent_weight_partitions, inds_partitions = partition_data(thetas, freq, psi, sent_weights, partition_size=50)

        partitions = zip(theta_partitions, freq_partitions, psi_partitions, sent_weight_partitions, inds_partitions)

        j = 0

        for theta_partition, freq_partition, psi_partition, sent_weight_partition, inds_partition in partitions:

            j += 1

            for _ in range(parameter_iterations):

                print "Epoch: {} | partition {}/{} | iteration {} over R".format(i,j,len(theta_partition), _)

                cost, grad_wrt_R   = get_dcostdR(R.astype('float32'), theta_partition.astype('float32'), freq_partition.astype('float32'), theta_reg_weight, frobenius_reg_weight, psi_partition.astype('float32'), sent_weight_partition.astype('float32'))
                cost, grad_wrt_psi = get_dcostdpsi(R.astype('float32'), theta_partition.astype('float32'), freq_partition.astype('float32'), theta_reg_weight, frobenius_reg_weight, psi_partition.astype('float32'), sent_weight_partition.astype('float32'))

                # don't want to update the first 2 rows of psi due to bias terms
                psi_partition[2:,:] += learning_rate * grad_wrt_psi[2:,:]
                R += learning_rate * grad_wrt_R

                if old_cost is None:
                    old_cost = cost
                elif converges(old_cost, cost):
                    break
                else:
                    # print "change in cost wrt R: ", old_cost - cost
                    old_cost = cost

            psi = update_parameter(psi, psi_partition, inds_partition)

            # converges a lot faster. so just wait until it reaches a cost change of zero.
            old_cost = None
            for _ in range(parameter_iterations):

                print "Epoch: {} | partition {}/{} | iteration {} over theta".format(i,j,len(theta_partition), _)

                cost, grad_wrt_theta = get_dcostdtheta(R.astype('float32'), theta_partition.astype('float32'), freq_partition.astype('float32'), theta_reg_weight, frobenius_reg_weight, psi_partition.astype('float32'), sent_weight_partition.astype('float32'))

                grad_wrt_theta[2:,:] *= learning_rate

                # don't want to update the first two rows of the theta matrix due to bias terms
                theta_partition[2:,:] += grad_wrt_theta[2:,:]

                if old_cost is None:
                    old_cost = cost
                elif converges(old_cost, cost):
                    break
                else:
                    # print "change in cost wrt theta: ", old_cost - cost
                    old_cost = cost

            thetas = update_parameter(thetas, theta_partition, inds_partition)

    return R

if __name__ == "__main__":


#    freq = numpy.random.randint(low=0,high=10,size=(5,5))
#    theta,R = create_parameters(5,5,5)

#    theta_samples, freq_samples, inds = sample_data(theta,freq)


#    print theta_samples
#    print freq_samples
#    print inds

#    grad_update = numpy.random.randn(6,5)

#    theta1 = update_parameter(theta, grad_update, inds)

#    print theta1

#    print theta

#    exit()

    # large example. should work!
#    freq = numpy.random.randint(low=0,high=10,size=(5,5))
#    theta,R = create_parameters(5,5,5)

    # small corpus example
    docs = 20
    words = 5000
    size = 50

    # tiny example
    # docs = 25
    # words = 500
    # size = 15

    freq = numpy.random.randint(low=0,high=10,size=(words,docs))
    sentiment_weights = get_sentiment_weights(docs/2, docs/4, docs/4)

    theta,R,psis = create_parameters(size,words,docs)

    # freq = numpy.random.randint(low=0,high=10,size=(5000,25000))
    # theta,R,psi = create_parameters(50,5000,25000)

    # freq = numpy.random.randint(low=0,high=10,size=(5,5))
    # theta,R = create_parameters(5,5,5)

    gradient_ascent(R.astype('float32'), theta.astype('float32'), freq.astype('float32'), psis.astype('float32'), sentiment_weights.astype('float32'))

    pass


