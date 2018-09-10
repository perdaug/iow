
"""
VERSION
- Python 3

FUNCTION
- Stem the lexicon content

NOTES
- Write the corpora of interest into the configuration file
"""

import os
from stemming.porter2 import stem
from optparse import OptionParser
from nltk.tokenize import RegexpTokenizer
import pandas as pd

op = OptionParser()
op.add_option('--lexicon', action='store', type=str,
              help='The source of the lexicon.')
op.add_option('--claim', action='store', type=str,
              help='The lexicon subset.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_DATA = PATH_HOME + '/data/DP/text/' + opts.lexicon + '/'
PATH_OUT = PATH_HOME + '/data/FE/textual/' + opts.lexicon + '/'
PATH_SETTINGS = PATH_HOME + '/data/settings/'
# ___________________________________________________________________________


def main():
    tokenizer = RegexpTokenizer(r'\w+')
    '''
    Getting the claim directory names.
    '''
    name_dirs_file = 'dirs_' + opts.claim + '.txt'
    with open(PATH_SETTINGS + name_dirs_file, 'r') as file_in:
        string_dirs_claim = file_in.read()[:-1]
    dirs_claim = string_dirs_claim.split('\n')
    '''
    Stemming the claim corpora.
    '''
    for corpus in dirs_claim:
        path_corpus_in = PATH_DATA + corpus + '/'
        path_corpus_out = PATH_OUT + 'features-separate_' + opts.claim + '/' \
            + corpus + '/'
        if not os.path.exists(path_corpus_out):
            os.makedirs(path_corpus_out)
        '''
        - Stemming the file content
        - Appending the stemmed content
        '''
        # out_string = ''
        for name_file in os.listdir(path_corpus_in):
            print(name_file)
            feature_text = pd.read_pickle(path_corpus_in + name_file)
            feature_tokenised = tokenizer.tokenize(feature_text)
            tokens_stemmed = []
            for token in feature_tokenised:
                token_stemmed = stem(str(token))
                tokens_stemmed.append(token_stemmed)
            feature_merged = ' '.join(tokens_stemmed)
            # out_string += ' ' + feature_merged
            name_out_file = opts.lexicon + '_' + name_file + '.txt'
            with open(path_corpus_out + name_out_file, 'w') as file_out:
                file_out.write(feature_merged)


if __name__ == '__main__':
    main()
