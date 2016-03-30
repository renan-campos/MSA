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

def main():
  training_set = data.train['unsup'] + data.train['pos'] + data.train['neg']

  # Print some stats about the training set
  print "### Training set stats: ###\n\
  %d positive reviews\n\
  %d negative reviews\n\
  %d unannoteded reviews" % (len(data.train['pos']), 
                             len(data.train['neg']), 
                             len(data.train['unsup']))

  # Create the vectorizer that builds a vector representation of the data.
  if not vectorizer.load_vecs():
    vectorizer.set_vocab(training_set, 5000, 50)
  vectorizer.dump_vecs()

  #TODO Train the R vector

  #TODO Use R vectors to calculate semantic score

  #TODO Feed the semantic score to SVM

if __name__ == '__main__':
  main()
