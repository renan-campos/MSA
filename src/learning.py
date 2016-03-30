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
    # ***************************************************************************************************
    # ************NOTE: make sure we cut the bias off when computing final result...*********************
    # ***************************************************************************************************
    R    = numpy.random.randn(B,V)
    bias = numpy.random.randn(1,V)

    thetas = numpy.random.randn(B,D)
    ones  = numpy.ones((1,D))

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

# TODO: currently verifying my vectorized implementation by hand....
def gradient_wrt_R_ij(i,j,R,thetas):
    """
        computes gradient w.r.t (i,j) element within matrix R

        indices (i,j) are base 1

        gradient is of the form:

            for each document (doc_k) and each word (w) in doc_k, sum the following:

                (d/A[i,j] theta_k^T * phi_w + b_w) - (d/A[i,j] sum_of_w'_in_V(theta_k^T*phi_w' + b_w'))

                for doc_k's theta vector (theta_k) if word's (w) col (j) matches col in (i,j) parameter then:
                    (d/A[i,j] theta_k^T * phi_w + b_w) = i'th element of theta_k vector
                else:
                    (d/A[i,j] theta_k^T * phi_w + b_w) = 0

                (d/A[i,j] sum_of_w'_in_V(theta_k^T*phi_w' + b_w') =
                    (1 / log(sum of -E(w')) * sum_of_all_words_w'_in_V(exp(-E(w')* (d/A[i,j] theta_k^T * phi_w' + b_w')))

        Did my best to vectorize the above derivation of the gradient.
    """

    i = i - 1
    j = j - 1

    # one hot matrix. B has all zero entries except at i,j
    B = numpy.zeros(R.shape)
    B[i][j] = 1

    # row = word_i, col = doc_k
    B_t = numpy.transpose(B)

    # energies of words in a document.
    # row = word_i, col = doc_k
    Energies = -E_all(R,thetas)

    # vector of real-values
    coef = (1 / numpy.log(numpy.sum(numpy.exp(Energies),0)))

    return sum(sum(numpy.dot(B_t,thetas) - coef * sum(Energies * numpy.dot(B_t,thetas))))


#def gradient_wrt_theta_k(j, ):
#    """
#        computes gradient w.r.t theta_j element within theta vector document.
#    """

#    if


if __name__ == "__main__":

    theta,R = create_parameters(1,2,2)

#    print theta
#    print R

#    print E([0,1],theta,R)

    print gradient_wrt_R_ij(1,1,R,theta)

    pass





