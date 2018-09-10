
"""
VERSION
- Python 3

FUNCTION
- Extract the images with the names.
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
PATH_DATA = PATH_HOME + '/data/IR/fb/images/raw/' + opts.query + '/'
PATH_OUT = PATH_HOME + '/data/IR/fb/images/processed/' + opts.query + '/'

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def main():
    ctr_total = 0
    ctr = 0
    for name_image_file in os.listdir(PATH_DATA):
        '''
        Extracting the image ids and names.
        '''
        # print('Processing: ' + name_image_file)
        images_extracted = []
        image_file = pd.read_pickle(PATH_DATA + name_image_file)
        for batch_image in image_file:
            for image in batch_image:
                image_extracted = {}
                ctr_total += 1
                if 'name' in image:
                    image_extracted['name'] = image['name']
                    image_extracted['id'] = image['id']
                    images_extracted.append(image_extracted)
                    ctr += 1
        pkl.dump(images_extracted, open(PATH_OUT + name_image_file, 'wb'))
    print('The image data extraction is finished. (%d/%d)'
          % (ctr, ctr_total))


if __name__ == '__main__':
    main()
