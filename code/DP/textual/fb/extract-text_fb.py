
"""
VERSION
- Python 3

FUNCTION
- Extract the textual image data.
"""

import pandas as pd
import pickle as pkl
import os
from optparse import OptionParser

op = OptionParser()
op.add_option('--query', action='store', type=str, help='The image class.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_DATA = PATH_HOME + '/data/IR/fb/images/processed/' + opts.query + '/'
PATH_OUT = PATH_HOME + '/data/DP/text/fb/' + opts.query + '/'

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________
# TODO: The album retrieval had a description, consider taking that info


def main():
    ctr = 0
    for batch_image in os.listdir(PATH_DATA):
        batch_image = pd.read_pickle(PATH_DATA + batch_image)
        for image in batch_image:
            name_image = image['id'] + '.pkl'
            print('Writing: ' + name_image)
            pkl.dump(image['name'], open(PATH_OUT + name_image, 'wb'))
            ctr += 1
    print('Total image names retrieved: %d' % ctr)


if __name__ == '__main__':
    main()
