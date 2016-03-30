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

def gradient_wrt_R_ij(i,j,R,thetas,freq_matrix):

    # TODO: compute gradient of documents one at a time. computing multiple at once is not practicle

    """
        TODO: make a latex file of derivation for peer review and add to repo. this notation is hard to read.
        TODO: I tested some examples by hand, it is possible there are some mistakes.
        TODO: there are way too many files for what is here. need to do it all over again...

        freq_matrix should be a numpy matrix where the col represents the document and the row represents
        how many times a word occurs within the document.

        computes gradient w.r.t (i,j) element within matrix R

        indices (i,j) are base 1

        gradient is of the form:

            for each document (doc_k) and each word (w) in doc_k, sum the following:

                (d/R[i,j] theta_k^T * phi_w + b_w) - (d/R[i,j] sum_of_w'_in_V(theta_k^T*phi_w' + b_w'))

                for doc_k's theta vector (theta_k) if word's (w) col (j) matches col in (i,j) parameter then:
                    (d/R[i,j] theta_k^T * phi_w + b_w) = i'th element of theta_k vector
                else:
                    (d/R[i,j] theta_k^T * phi_w + b_w) = 0

                (d/R[i,j] sum_of_w'_in_V(theta_k^T*phi_w' + b_w') =
                    (1 / log(sum of -E(w')) * sum_of_all_words_w'_in_V(exp(-E(w')* (d/R[i,j] theta_k^T * phi_w' + b_w')))

        Did my best to vectorize the above derivation of the gradient.
    """

    global B_t

    i = i - 1
    j = j - 1

    # one hot matrix. B has all zero entries except at i,j
    # row = word_i, col = doc_k
    if B_t is None:
        B_t = numpy.zeros((R.shape[1],R.shape[0]))
    B_t[j][i] = 1

    # energies of words in vocbulary.
    # row = word_i, col = doc_k
    Energies = -E_all(R,thetas)

    e_E = numpy.exp(Energies)
    df_E = numpy.dot(B_t,thetas)

    # vector of real-values
    coef = (1/numpy.log(numpy.sum(e_E,0)))

    B_t[j][i] = 0

    return numpy.sum(numpy.sum((df_E - (coef * numpy.sum(e_E * df_E))) * freq_matrix))


#def gradient_wrt_theta_k(j, ):
#    """
#        computes gradient w.r.t theta_j element within theta vector document.
#    """

#    if


if __name__ == "__main__":

#    theta,R = create_parameters(20,50,100)

#    gradient_wrt_R_ij(1,1,R,theta,numpy.random.randn(50,100))

    theta,R = create_parameters(50,5000,25000)

    init_time = time.time()

    print R.shape

    freq_mat = numpy.random.randn(5000,25000)

    for i in range(0,R.shape[0]):
        i += 1
        for j in range(0,R.shape[1]):

            j += 1

            # ex: creating an example frequence matrix
            gradient_wrt_R_ij(i,j,R,theta,freq_mat)

    print time.time() - init_time

    pass


