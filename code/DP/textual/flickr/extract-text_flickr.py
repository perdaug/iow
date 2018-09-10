
"""
VERSION
- Python 3

FUNCTION
- Extract the textual image data.
"""

import multiprocessing
import requests
import os
from optparse import OptionParser
import pandas as pd
import pickle as pkl

op = OptionParser()
op.add_option('--query', action='store', type=str, help='The image class.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow/'
PATH_DATA = PATH_HOME + 'data/IR/flickr/ids_image/'
PATH_OUT = PATH_HOME + 'data/DP/text/flickr/' + opts.query + '/'
KEY_API = 'b1daebefe4cdd0a9ce5945683be0c463'
N_THREADS = 16

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def retrieve_image(lock, ns, id_image):
    try:
        lock.acquire()
        ns.counter_total += 1
        lock.release()
        name_file = id_image + '.pkl'
        if os.path.exists(PATH_OUT + name_file):
            print('The file already exists: %s' % id_image)
            return
        '''
        Retrieving the image info.
        '''
        print('Retrieving: %s' % (id_image))
        url_image_info = 'https://api.flickr.com/services/rest/' \
                         '?method=flickr.photos.getInfo' \
                         '&format=json&nojsoncallback=?' \
                         '&api_key=%s&photo_id=%s' \
                         % (KEY_API, id_image)
        response_image = requests.get(url_image_info)
        image = response_image.json()
        '''
        Extracting the textual features.
        '''
        tags_image = image['photo']['tags']['tag']
        title_image = image['photo']['title']
        description_image = image['photo']['description']
        textual_feature = ''
        for tag in tags_image:
            if '_content' in tag:
                textual_feature += tag['_content']
        if '_content' in title_image:
            textual_feature += title_image['_content']
        if '_content' in description_image:
            textual_feature += description_image['_content']
        pkl.dump(textual_feature, open(PATH_OUT + name_file, 'wb'))
        print('Writing: %s' % id_image)
        lock.acquire()
        ns.counter += 1
        lock.release()
    except Exception as e:
        print(e)
        return

# ___________________________________________________________________________


def main():
    name_data_file = 'ids_' + opts.query + '.pkl'
    dict_image = pd.read_pickle(PATH_DATA + name_data_file)
    ids_image = dict_image['data'].keys()
    '''
    Enabling multiprocessing.
    '''
    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    ns.counter = 0
    ns.counter_total = 0
    lock = manager.Lock()
    processes = multiprocessing.Pool(N_THREADS)
    from functools import partial
    func = partial(retrieve_image, lock, ns)
    processes.map(func, ids_image)
    print('The meta-data retrieval is finished. (%d/%d)'
          % (ns.counter, ns.counter_total))


if __name__ == '__main__':
    main()
