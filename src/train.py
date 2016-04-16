"""
 Text-Machine Lab: MSA

 File Name : train.py

 Creation Date : 29-03-2016

 Created By : Renan Campos

 Purpose : Trains a support vector machine to classify positive/negative
           movie reviews based on positive, negative, and unnanotated data.
           Current features:
            * Semantic similarity between reviews.

"""

import vectorizer
import data
import learning

from sklearn import svm
import numpy as np

import os
import sys
import cPickle as pickle

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

POS = 1
NEG = 0

BETA = 50

ITER = 40
LAMBDA = 1e-4

# NOTE: Delete pickled vectorizers after changing these
VOCAB  = 5000
IGNORE = 50

def main():
  training_set = data.train['unsup'] + data.train['pos'] + data.train['neg']

  # Print some stats about the training set
  print "### Training set stats: ###\n\
  %d positive reviews\n\
  %d negative reviews\n\
  %d unannoteded reviews" % (len(data.train['pos']),
                             len(data.train['neg']),
                             len(data.train['unsup']))
  """
  d = list()
  for doc in training_set:
    d.append(doc)
    if len(d) > 5:
      break
  training_set = d
  """

  # Create the vectorizer that builds a vector representation of the data.
  #if not vectorizer.load_vecs():
  vectorizer.set_vocab(training_set, VOCAB, IGNORE)
  vectorizer.dump_vecs()

  freqs = vectorizer.bow_vecs(training_set)

  freqs = np.transpose(freqs)

  actual_vocab_size = freqs.shape[0]

  # Train the R vector
  if actual_vocab_size < VOCAB:
    thetas,R = learning.create_parameters(BETA,actual_vocab_size,len(training_set))
  else:
    thetas,R = learning.create_parameters(BETA, VOCAB, len(training_set))

  """
  print "len(d): ", len(d)
  print "R.shape: ",R.shape
  print "thetas.shape: ", thetas.shape
  print "freqs.shape: ", freqs.shape
  """

  print "learning Vectors..."
  R = learning.gradient_ascent(R, thetas, freqs, iterations=ITER, learning_rate=LAMBDA)

  X = list()
  Y = list()
  print "tagging examples..."
  for review in data.train['pos']:
    X.append(learning.phi(R[1:], vectorizer.tfidf_bow(review).T).flatten())
    Y.append(POS)
  for review in data.train['neg']:
    X.append(learning.phi(R[1:], vectorizer.tfidf_bow(review).T).flatten())
    Y.append(NEG)

  X = np.array(X)

  # Train SVM
  print "training model..."
  clf = svm.SVC()
  clf.fit(X,Y)

  # Pickle svm to file
  print "dumping trained model and vectors..."
  with open(os.path.join(TMP_DIR, 'svm.pickle'), 'wb') as f:
    pickle.dump(clf, f)
  # Pickle R matrix to file
  with open(os.path.join(TMP_DIR, 'R.pickle'), 'wb') as f:
    pickle.dump(R, f)

if __name__ == '__main__':
  main()
