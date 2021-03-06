"""                                                                              
 Text-Machine Lab: MSA

 File Name : predict.py

 Creation Date : 30-03-2016

 Created By : Connor Cooper

 Purpose : Predicts the positive/negative polarity of the development set, reports
           precision, recall, f1, and accuracy; also writes the predictions 
           to files for further error analysis.
"""

import vectorizer
import data
import learning


import os
import sys
import cPickle as pickle
import numpy as np

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

POS = 1
NEG = 0

def main(dev, out=sys.stdout):
  test_set = dev['pos'] + dev['neg']

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
  # Y is the target class
  Y = list()
  for review in dev['pos']:
    X.append(learning.phi(R[1:], vectorizer.tfidf_bow(review).T).flatten())
    Y.append(POS)
  for review in dev['neg']:
    X.append(learning.phi(R[1:], vectorizer.tfidf_bow(review).T).flatten())
    Y.append(NEG)

  X = np.array(X) 

  # Load SVM classifier and classify test cases.
  with open(os.path.join(TMP_DIR, 'svm.pickle'), 'rb') as f:
    clf = pickle.load(f)

  # H is the list of hypotheses.
  H = clf.predict(X).tolist()

  # Process output
  # Calculates precision, recall, f1 score, and accuracy. 
  # Prints lists of true positives, false positives, true negatives and 
  # false negatives to individual files for error analysis.
  TP = list()
  TN = list()
  FP = list()
  FN = list()

  for review, target, prediction in zip(test_set, Y, H):
    if (target == prediction and target == POS):
      TP.append(review.file)
    elif (target == prediction and target == NEG):
      TN.append(review.file)
    elif (target != prediction and target == POS):
      FN.append(review.file)
    else:
      FP.append(review.file)

  with open(os.path.join(TMP_DIR, 'TP.txt'), 'w') as f:
    for each in TP:
      f.write(each)
      f.write('\n')
  with open(os.path.join(TMP_DIR, 'TN.txt'), 'w') as f:
    for each in TN:
      f.write(each)
      f.write('\n')
  with open(os.path.join(TMP_DIR, 'FP.txt'), 'w') as f:
    for each in FP:
      f.write(each)
      f.write('\n')
  with open(os.path.join(TMP_DIR, 'FN.txt'), 'w') as f:
    for each in FN:
      f.write(each)
      f.write('\n')
      
  precision = len(TP) / float(len(TP)+len(FP))
  recall    = len(TP) / float(len(TP)+len(FN))
  f1        = (2*precision * recall) / (precision + recall)
  accuracy  = (len(TP) + len(TN)) / float(len(TP) + len(TN) + len(FP) + len(FN))
  print >>out, "*** Results ***"
  print >>out, "\tTotal dev files: %d" % (len(test_set))
  print >>out, "\t(%d positive, %d negative)" % (len(dev['pos']), len(dev['neg']))
  print >>out, " "
  print >>out, "\tTP: %d TN: %d FP: %d FN: %d" % (len(TP), len(TN), len(FP), len(FN)) 
  print >>out, " "
  print >>out, "\tPrecision: %0.4f" % (precision)
  print >>out, "\tRecall: %0.4f" % (recall)
  print >>out, "\tF1: %0.4f" % (f1)
  print >>out, "\tAccuracy: %0.4f" % (accuracy)


if __name__ == "__main__":
    main(data.train)
