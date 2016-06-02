"""                                                                              
 MSA  

 File Name : data.py
                                                                              
 Creation Date : 29-02-2016
                                                                              
 Created By : Renan Campos                                               
                                                                              
 Purpose : This module defines a data representation for the review files.
           Also builds dictionaries containing training/test data: 
    
    The train variable is set up as follows:
                |-> neg   -> set(Reviews)
      train {} -|-> pos   -> set(Reviews)
    
    The test variable is set up as follows:
               |-> neg -> set(Reviews)
      test {} -|-> pos -> set(Reviews)

    A Review is contains the id, rank and text of a review file.
    
    The naming of the review files are set up as follows:
      ID_RANK.txt

"""

import os
import sys

from collections import defaultdict 

TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'train')
TEST_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'test')

#
# data.train
# data.test

class Review:
  """
    Representation of a review file:
      id   - review id
      rank - review ranking
      text - contents of the review
  """

  def __init__(self, filename):
    self.id   = os.path.basename(filename).split(".")[0].split("_")[0]
    self.file = filename

  def getText(self):
    return open(self.file, 'r').read()

if not os.path.isdir(TRAIN_DIR):
  sys.stderr.write( \
  "ERROR: Training directory not found. Please do the following:\n\
  \t$ mkdir data/train/pos data/train/neg\n")
  sys.exit(1)
train = defaultdict(list)
for type in ('pos', 'neg'):
  for file in os.listdir(os.path.join(TRAIN_DIR, type)):
    if file.endswith('txt'):
     train[type].append(Review(os.path.join(TRAIN_DIR, type, file)))

if not os.path.isdir(TEST_DIR):
  sys.stderr.write( \
  "ERROR: Testing directory not found. Please do the following:\n\
  \t$ mkdir data/test\n")
  sys.exit(1)

test = list()
for file in os.listdir(TEST_DIR):
  if file.endswith('txt'):
    test.append(Review(os.path.join(TEST_DIR, file)))
test.sort(key=lambda x: int(x.id))

