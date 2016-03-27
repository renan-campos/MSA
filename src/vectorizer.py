"""                                                                              
 Text-Machine Lab: MSA 

 File Name : vectorizer.py
                                                                              
 Creation Date : 26-03-2016
                                                                              
 Created By : Renan Campos                                               
                                                                              
 Purpose : This module creates a tf-idf weighted Bag-of-Words vector 
           representation of the given text. 
           Optional filtering includes:
           - Only include N most frequent words in the corpus
           -Ignore M most frequent.

"""

import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

from tokenizer import tokenize

VECTORIZER = None

def set_vocab(docs, N, M):
  """
    Creates the vocabulary which the vectorizer will use to filter.
    docs - a set of reviews
    N - N most frequent words in the corpus
    M - Ignore M most frequent

    This also teaches the tfidf vectorizer the idf vector.
  """
  
  global VECTORIZER

  V =  CountVectorizer(preprocessor=(lambda x: x.getText()), tokenizer=tokenize)

  # Create a document-term matrix of frequencies 
  F = V.fit_transform(docs)
  F = F.toarray()
  
  # The terms in the same order as matrix Freqs
  T = V.get_feature_names()
  
  # Indexing example
  # >>> a = [1,2,3,4,5,6,7,8]
  # >>> # I want the last 5 not including the last 3
  # ... a[-5:][:-3]
  # [4, 5]


  # Sort terms by total frequency, and filter.
  vocab = set([(t) for (f,t) in sorted(zip(np.sum(F, axis=0), T))][-(N+M):][:-M])

  # Build the tfidf vectorizer
  VECTORIZER = TfidfVectorizer(
                                preprocessor=(lambda x: x.getText()), 
                                tokenizer=tokenize,
                                vocabulary=vocab)
  VECTORIZER.fit(docs)


def tfidf_bow(doc):
  """
    Returns a "term-frequency inverse-document-frequency" weighted bag of words
    vector from the given document.
  """

  if VECTORIZER == None:
    sys.stderr.write("ERROR: Vectorizer not defined... Did you call set_vocab?\n")
    return None
  return VECTORIZER.transform([doc])

def main():
  import data

  X = list()
  for i in range(5):
    X.append(data.train['unsup'].pop())

  # Set the vocab to be the 5 most frequent words (ignoring first 10)
  set_vocab(X, 5, 10)

  w = tfidf_bow(data.train['unsup'].pop())
  
  print VECTORIZER.get_feature_names()
  print w.toarray()[0]

if __name__ == '__main__':
  main()
