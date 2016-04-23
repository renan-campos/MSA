"""                                                                              
 Text-Machine Lab: MSA  

 File Name : msa.py
                                                                              
 Creation Date : 21-04-2016
                                                                              
 Created By : Renan Campos                                               
                                                                              
 Purpose : Main script. Runs in batches. 

"""

import data
import vectorizer

import train
import predict
import learning

import os
import cPickle as pickle
from collections import defaultdict

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

if __name__ == '__main__':
  
  training_set = data.train['unsup'] + data.train['pos'] + data.train['neg']
  
  print "Creating vectorizer"
  vectorizer.set_vocab(training_set, 5000, 50)
  vectorizer.dump_vecs()
  
  thetas,R,psis = learning.create_parameters(50, 5000, 600)
  # Pickle theta, R and psis to files
  with open(os.path.join(TMP_DIR, 'thetas.pickle'), 'wb') as f:
    pickle.dump(thetas, f)
  with open(os.path.join(TMP_DIR, 'R.pickle'), 'wb') as f:
    pickle.dump(R, f)
  with open(os.path.join(TMP_DIR, 'psis.pickle'), 'wb') as f:
    pickle.dump(psis, f)


  print "Batch loop - running train and predict" 
  for i in range(125):

    print "****** Batch #%d ******" % (i+1)

    # Build training set
    t = defaultdict(list)
    for j in range(400):
      t['unsup'].append(data.train['unsup'].pop())
    for j in range(100):
      t['pos'].append(data.train['pos'].pop())
    for j in range(100):
      t['neg'].append(data.train['neg'].pop())

    # train
    train.main(t)

    with open(os.path.join(TMP_DIR, 'batch_%d.txt' % (i)), 'w') as f:
      # predict
      predict.main(f)

