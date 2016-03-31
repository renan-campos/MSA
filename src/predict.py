"""                                                                              
 Text-Machine Lab: MSA

 File Name : predict.py

 Creation Date : 30-03-2016

 Created By : Connor Cooper

 Purpose : Predicts the positive/negative polarity of a movie review using
           a pre-trained model
"""

import vectorizer
import data

def main():
    test_set = data.test['pos'] + data.test['neg']

    # load vectorizer
    if not vectorizer.load_vecs():
        print "Vectorizer not found: Train a model before attempting to predict"
        pass

    # get word vectors
    words = []
    for doc in test_set:
        words.append(vectorizer.onehot_vecs(doc))

    # TODO: get semantic score from R matrix (saved somewhere)

    # TODO: pass semantic score to SVM classifier

    # TODO: process output (print file name an classification to stdout, for example)

if __name__ == "__main__":
    main()
