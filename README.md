# MSA
Movie Sentiment Analyzer

###Dependencies:

    - python (2.7.6)
        - Numpy (1.11.0)
        - Scipy (0.17.0)
        - Theano (0.9.0)
        - Sklearn (0.17.1)

###Installation:

    1. Install above python dependencies.
        - We recommend using a virtual environment (virtualenv) along with pip to install dependendencies. Bottomline, make sure dependencies are importable when running python.
        - Make sure the proper versions of the dependencies are installed, we do not guarantee it will work otherwise.
    2. Clone repository.
        $ git clone https://github.com/renan-campos/MSA.git

###How to train competition model:

    # make sure working directory is the cloned directory
    $ python src/train.py

###How to predict on competition data:

    # make sure working directory is the cloned directory
    $ python src/test.py

