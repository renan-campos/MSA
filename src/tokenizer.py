"""                                                                              
 Text-Machine Lab: MSA 

 File Name : tokenizer.py
                                                                              
 Creation Date : 26-03-2016
                                                                              
 Created By : Renan Campos                                               
                                                                              
 Purpose : A sentiment tokenizer based off of the tutorial:
             http://sentiment.christopherpotts.net/tokenizing.html

           This tokenizer is able to handle:
           * Emoticons
           * Punctuation (Word-internal marks included)
           * Lower-case to reduce sparsity 
"""

#
#dd Tokenizer. Code based off of: http://sentiment.christopherpotts.net/tokenizing.html
#

import re

# The components of the tokenizer:
regex_strings = (
  # Emoticons:
  r"""
  (?:
  [<>]?                      # eybrows
  [:;=8]                     # eyes
  [\-o\*\']?                 # optional nose
  [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
  |
  [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
  [\-o\*\']?                 # optional nose
  [:;=8]                     # eyes
  [<>]?
  )""",    
  # Remaining word types:
  r"""
  (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
  |
  (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
  |
  (?:[\w_]+)                     # Words without apostrophes or dashes.
  |
  (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
  |
  (?:\S)                         # Everything else that isn't whitespace.
  """
  )

WORD_RE = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon string gets its own regex to preserve case.
EMO_RE = re.compile(regex_strings[0], re.VERBOSE | re.I | re.UNICODE)

def tokenize(s):
  words = WORD_RE.findall(s)

  # Remove case (as it leads to unnecessary sparseness, but keep case for emoticons :D)
  words = map(( lambda x : x if EMO_RE.search(x) else x.lower()), words)

  return words
