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
import predict

from sklearn import svm
import numpy as np

import os
import sys
import cPickle as pickle

from collections import defaultdict
from random import shuffle

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

POS = 1
NEG = 0

BETA = 50

ITER = 40
LAMBDA = 1e-4

# NOTE: Delete pickled vectorizers after changing these
VOCAB  = 5000
IGNORE = 50

# Development and batch size variables
DEV_SIZE = 500
BATCHES = 40
BATCH_SIZE = 600


def main(t = None):
  
  if t:
    training_set = t['pos'] + t['neg']
  else:
    training_set = data.train['pos'] + data.train['neg']

  # Print some stats about the training set
#  print "### Training set stats: ###\n\
#  %d positive reviews\n\
#  %d negative reviews\n\
#  %d unannoteded reviews" % (len(data.train['pos']),
#                             len(data.train['neg']),
#                             len(data.train['unsup']))

  # d = list()
  # for doc in training_set:
  #   d.append(doc)
  #   if len(d) > 50:
  #     break
  # training_set = d

  # Create the vectorizer that builds a vector representation of the data.
  #if not vectorizer.load_vecs():
  if t:
    vectorizer.load_vecs()
  else:
    vectorizer.set_vocab(training_set, VOCAB, IGNORE)
    vectorizer.dump_vecs()

  freqs = vectorizer.bow_vecs(training_set)

  freqs = np.transpose(freqs)

  actual_vocab_size = freqs.shape[0]

  # Train the R vector
  if t:
    with open(os.path.join(TMP_DIR, 'thetas.pickle'), 'rb') as f:
      thetas = pickle.load(f)
    with open(os.path.join(TMP_DIR, 'R.pickle'), 'rb') as f:
      R = pickle.load(f)
    with open(os.path.join(TMP_DIR, 'psis.pickle'), 'rb') as f:
      psis = pickle.load(f)
  else:
    if actual_vocab_size < VOCAB:
      thetas,R,psis = learning.create_parameters(BETA,actual_vocab_size,len(training_set))
    else:
      thetas,R,psis = learning.create_parameters(BETA, VOCAB, len(training_set))

  """
  print "len(d): ", len(d)
  print "R.shape: ",R.shape
  print "thetas.shape: ", thetas.shape
  print "freqs.shape: ", freqs.shape
  """
  if t:
    sentiment_weights = learning.get_sentiment_weights(0, len(t['pos']), len(t['neg']))
  else:
    sentiment_weights = learning.get_sentiment_weights(0, len(data.train['pos']), len(data.train['neg']))
    # sentiment_weights = sentiment_weights[:, 0:51]
    

  print "learning vectors..."
  R = learning.gradient_ascent(R.astype('float32'), thetas.astype('float32'), freqs.astype('float32'), psis.astype('float32'), sentiment_weights.astype('float32'), iterations=ITER, learning_rate=LAMBDA)

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
  # Pickle theta, R and psis to files
  with open(os.path.join(TMP_DIR, 'thetas.pickle'), 'wb') as f:
    pickle.dump(thetas, f)
  with open(os.path.join(TMP_DIR, 'R.pickle'), 'wb') as f:
    pickle.dump(R, f)
  with open(os.path.join(TMP_DIR, 'psis.pickle'), 'wb') as f:
    pickle.dump(psis, f)

#
# Main
#
if __name__ == '__main__':
  """
    The R Matrix is trained by using cross-validation against a development set.
  """

  # Shuffling training data
  shuffle(data.train['pos'])
  shuffle(data.train['neg'])

  training_set = data.train['pos'] + data.train['neg']
  
  print "Creating vectorizer"
  vectorizer.set_vocab(training_set, VOCAB, IGNORE)
  vectorizer.dump_vecs()
  
  thetas,R,psis = learning.create_parameters(BETA, VOCAB, 600)
  # Pickle theta, R and psis to files
  with open(os.path.join(TMP_DIR, 'thetas.pickle'), 'wb') as f:
    pickle.dump(thetas, f)
  with open(os.path.join(TMP_DIR, 'R.pickle'), 'wb') as f:
    pickle.dump(R, f)
  with open(os.path.join(TMP_DIR, 'psis.pickle'), 'wb') as f:
    pickle.dump(psis, f)


  print "Creating a development set for cross-validation"
  dev = defaultdict(list)

  for i in range(DEV_SIZE):
    dev['pos'].append(data.train['pos'].pop())
    dev['neg'].append(data.train['neg'].pop())

  pos = data.train['pos'][:]
  neg = data.train['neg'][:]

  print "Batch loop - running train and predict" 
  for i in range(BATCHES):

    print "****** Batch #%d ******" % (i+1)

    # Build training set
    t = defaultdict(list)
    for j in range(BATCH_SIZE/2):
      t['pos'].append(pos.pop())
      t['neg'].append(neg.pop())

    # train
    main(t)

    with open(os.path.join(TMP_DIR, 'batch_%d.txt' % (i)), 'w') as f:
      # predict
      predict.main(dev, f)

