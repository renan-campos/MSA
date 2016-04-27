"""                                                                              
 Text-Machine Lab: MSA

 File Name : test.py

 Creation Date : 24-04-2016

 Created By : Renan Campos

 Purpose : Creates a csv (comma seperated values) file labeling the test set.
"""

import vectorizer
import data
import learning

import os
import sys
import csv
import cPickle as pickle
import numpy as np

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

POS = 1
NEG = 0

def main():

  # load vectorizer
  if not vectorizer.load_vecs():
    print "Vectorizer not found: Train a model before attempting to predict"
    return

  # load R matrix
  with open(os.path.join(TMP_DIR, 'R.pickle'), 'rb') as f:
    R = pickle.load(f)

  # get semantic scores for each review.
  # X is a list of input vectors
  X = list()
  for review in data.test:
    X.append(learning.phi(R[1:], vectorizer.tfidf_bow(review).T).flatten())

  X = np.array(X) 

  # Load SVM classifier and classify test cases.
  with open(os.path.join(TMP_DIR, 'svm.pickle'), 'rb') as f:
    clf = pickle.load(f)

  # H is the list of hypotheses.
  H = clf.predict(X).tolist()
  
  # Write predictions to a csv file.
  resfile = os.path.join(TMP_DIR, 'results.csv')
  with open(resfile, 'wb') as csvfile:
    out = csv.writer(csvfile)
    out.writerow(['id','labels'])
    out.writerows(zip([x.id for x in data.test], H))

  print "Testing complete! Results located in: %s" % (resfile)

if __name__ == "__main__":
    main()
