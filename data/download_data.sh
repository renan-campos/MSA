#! /bin/bash

#
# Downloads the imdb data needed to train/test MSA.
#

printf "Downloading data...\n"
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz > /dev/null

printf "Untaring data...\n"
tar -zxf aclImdb_v1.tar.gz > /dev/null

mv aclImdb/* ./
rmdir aclImdb

printf "Removing tarfile...\n"
rm aclImdb_v1.tar.gz

printf "Done.\n"
exit 0
