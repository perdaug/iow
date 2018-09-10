
"""
VERSION
- Python 3

FUNCTION
- Write image urls to files
"""

import os
import pandas as pd
import requests
import multiprocessing
from optparse import OptionParser
import pickle as pkl

op = OptionParser()
op.add_option('--clss', action='store', type=str,
              help='The image class (e.g., vietnam war).')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow/'
PATH_DATA = PATH_HOME + 'data/IR/flickr/ids_image/'
NAME_DATA_FILE = 'ids_' + opts.clss + '.pkl'
PATH_OUT = PATH_HOME + 'data/IR/flickr/urls_image/' + opts.clss + '/'
KEY_API = ''
N_THREADS = 16
SIZE_IMAGE = 7

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def retrieve_image(lock, ns, id_image):
    try:
        lock.acquire()
        ns.counter_total += 1
        lock.release()
        if os.path.exists(PATH_OUT + '/' + id_image + '.txt'):
            print('Exit: The file already exists: %s' % (id_image))
            return
        '''
        - Sending the image info request;
        - Extracting the image url.
        '''
        url_image_info = 'https://api.flickr.com/services/rest/' \
                         '?method=flickr.photos.getSizes' \
                         '&format=json&nojsoncallback=?' \
                         '&api_key=%s&photo_id=%s' \
                         % (KEY_API, id_image)
        response = requests.get(url_image_info)
        json_response = response.json()
        url_image = json_response['sizes']['size'][SIZE_IMAGE]['source']
        '''
        Writing the image url.
        '''
        info_image = {}
        info_image['url'] = url_image
        info_image['name'] = id_image + '.jpg'
        name_image_file = id_image + '.pkl'
        print('Writing: %s' % name_image_file)
        pkl.dump(info_image, open(PATH_OUT + name_image_file, 'wb'))
        lock.acquire()
        ns.counter += 1
        lock.release()
    except Exception as e:
        print(e)
        return
# ___________________________________________________________________________


def main():
    dict_ids = pd.read_pickle(PATH_DATA + NAME_DATA_FILE)
    ids_image = dict_ids['data'].keys()
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
    print('The retrieval is finished (%d/%d).'
          % (ns.counter, ns.counter_total))


if __name__ == '__main__':
    main()
