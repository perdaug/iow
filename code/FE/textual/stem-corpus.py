
"""
VERSION
- Python 3

FUNCTION
- Stem the textual features

DEPENDENCIES
- pip install stemming.
"""

import os
from stemming.porter2 import stem
from nltk.tokenize import RegexpTokenizer
from optparse import OptionParser
import pandas as pd

op = OptionParser()
op.add_option('--source',
              action='store', type=str,
              help='The source of corpora.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_DATA = PATH_HOME + '/data/DP/text/' + opts.source + '/'
PATH_OUT = PATH_HOME + '/data/FE/textual/' + opts.source + '/'

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def main():
    tokenizer = RegexpTokenizer(r'\w+')
    '''
    Going through the retrieved image classes individually.
    '''
    for class_image in os.listdir(PATH_DATA):
        ctr = 0
        ctr_total = 0
        path_class_out = PATH_OUT + class_image + '/'
        if not os.path.exists(path_class_out):
            os.makedirs(path_class_out)
        '''
        Preprocessing the textual features:
        - Tokenising
        - Stemming
        '''
        path_class = PATH_DATA + class_image + '/'
        for name_file in os.listdir(path_class):
            ctr_total += 1
            feature_text = pd.read_pickle(path_class + name_file)
            feature_tokenised = tokenizer.tokenize(feature_text)
            tokens_stemmed = []
            for token in feature_tokenised:
                token_stemmed = stem(str(token))
                tokens_stemmed.append(token_stemmed)
            feature_merged = ' '.join(tokens_stemmed)
            if not os.path.exists(path_class_out + name_file):
                '''
                Wring as a txt file for the succeeding work-flow step.
                '''
                with open(path_class_out + name_file, 'w') as file_out:
                    file_out.write(feature_merged)
                ctr += 1
        print('Finished processing %s. (%d/%d)'
              % (class_image, ctr, ctr_total))


if __name__ == '__main__':
    main()
