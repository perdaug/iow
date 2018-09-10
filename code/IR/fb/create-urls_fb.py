
"""
VERSION
- Python 3

FUNCTION
- Extract the image urls.
"""

import os
from optparse import OptionParser
import pickle as pkl
import pandas as pd

op = OptionParser()
op.add_option('--query',
              action='store', type=str,
              help='A particular collection of images.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_DATA = PATH_HOME + '/data/IR/fb/images/processed/' + opts.query + '/'
PATH_OUT = PATH_HOME + '/data/IR/fb/urls_image/' + opts.query + '/'

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def main():
    ctr = 0
    ctr_total = 0
    '''
    Creating the image urls.
    '''
    for image_file_name in os.listdir(PATH_DATA):
        batch_image = pd.read_pickle(PATH_DATA + image_file_name)
        for image in batch_image:
            ctr_total += 1
            url_image = 'https://graph.facebook.com/' \
                + image['id'] + '/picture?'
            info_image = {}
            info_image['url'] = url_image
            info_image['name'] = image['id'] + '.jpg'
            name_image_file = image['id'] + '.pkl'
            if not os.path.exists(PATH_OUT + name_image_file):
                pkl.dump(info_image, open(PATH_OUT + name_image_file, 'wb'))
                ctr += 1
    print('The image url extraction is finished. (%d/%d)'
          % (ctr, ctr_total))


if __name__ == '__main__':
    main()
