
"""
VERSION
- Python 3

FUNCTION
- Extracting the features from the albums.
"""

import os
from optparse import OptionParser
import pickle as pkl
import pandas as pd

op = OptionParser()
op.add_option('--query',
              action='store', type=str,
              help='Picture class.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_DATA = PATH_HOME + '/data/IR/fb/albums/raw/' + opts.query + '/'
PATH_OUT = PATH_HOME + '/data/IR/fb/albums/processed/' + opts.query + '/'

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def extract_data(album):
    try:
        fields = ['id', 'name']
        extracted_album = {}
        for field in fields:
            extracted_album[field] = album[field]
        return extracted_album
    except:
        return None
# ___________________________________________________________________________


def main():
    ctr = 0
    ctr_total = 0
    for name_album_batch_file in os.listdir(PATH_DATA):
        # print('Processing: ' + name_album_batch_file)
        batches_album = pd.read_pickle(PATH_DATA + name_album_batch_file)
        '''
        Extract the relevant album data.
        '''
        extracted_albums = []
        for batch_album in batches_album:
            for album in batch_album:
                ctr_total += 1
                extracted_album = extract_data(album)
                if extracted_album is not None:
                    extracted_albums.append(extracted_album)
                    ctr += 1
        pkl.dump(extracted_albums,
                 open(PATH_OUT + name_album_batch_file, 'wb'))
    print('Album data extraction is finished. (%d/%d)'
          % (ctr, ctr_total))


if __name__ == '__main__':
    main()
