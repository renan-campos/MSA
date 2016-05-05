# MSA
Movie Sentiment Analyzer

Downloading Data:
    - Go into MSA/data directory:
        ```
        $ cd <download_dir>/MSA/data
        ```
    - execute download script:
        ```
        $ bash download_data.sh
        ```

External Dependencies:

    - Numpy (1.11.0)
    - Scipy (0.17.0)
    - Theano (0.9.0)
    - Sklearn (0.17.1)
    - python (2.7.6)

Installation:

    1. Install above python dependencies.
        - We recommend using a virtual environment (virtualenv) along with pip to install dependendencies. Bottomline, make sure dependencies are importable when running python.
        - Make sure the proper versions of the dependencies are installed, we do not guarantee it will work otherwise.
    2. Clone repository.
        ```
        $ git clone https://github.com/renan-campos/MSA.git
        ```
    3. Download data (see above)

How to run and evaluate system:

    1. execute following:
        ```
        $ python train.py
        ```


