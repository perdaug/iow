"""
VERSION
- Python 3

FUNCTION
- Extracting the image ids from the album data.
"""

import os
import multiprocessing
import requests
from optparse import OptionParser
import pandas as pd
import pickle as pkl

op = OptionParser()
op.add_option('--query', action='store', type=str,
              help='Picture class.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_DATA = PATH_HOME + '/data/IR/fb/albums/processed/' + opts.query + '/'
PATH_OUT = PATH_HOME + '/data/IR/fb/images/raw/' + opts.query + '/'
ACCESS_TOKEN = ''
N_THREADS = 16

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def retrieve_image_batch(url):
    try:
        response_image_batch = requests.get(url)
        batch_image = response_image_batch.json()
        return batch_image
    except Exception as e:
        # print(e)
        return None
# ___________________________________________________________________________


def process_album(lock, ns, name_album_file):
    # print('Processing: ' + name_album_file)
    albums = pd.read_pickle(PATH_DATA + name_album_file)
    '''
    Retrieving the image batches.
    '''
    images = []
    print('Retrieving an album: {}'.format(name_album_file))
    for album in albums:
        lock.acquire()
        ns.counter_total += 1
        lock.release()
        id_album = album['id']
        url_image_batch = 'https://graph.facebook.com/' \
            '%s/?fields=photos&access_token=%s' \
            % (id_album, ACCESS_TOKEN)
        batch_image = retrieve_image_batch(url_image_batch)
        if batch_image is None or 'photos' not in batch_image:
            continue
        images.append(batch_image['photos']['data'])
        lock.acquire()
        ns.counter += 1
        lock.release()
        batch_image_next = batch_image['photos']['paging']
        while 'next' in batch_image_next:
            lock.acquire()
            ns.counter_total += 1
            lock.release()
            url_image_batch = batch_image_next['next']
            batch_image = retrieve_image_batch(url_image_batch)
            if batch_image is None or 'paging' not in batch_image:
                break
            batch_image_next = batch_image['paging']
            lock.acquire()
            ns.counter += 1
            lock.release()
            images.append(batch_image['data'])
    '''
    Writing the image ids.
    '''
    if not os.path.exists(PATH_OUT + name_album_file):
        pkl.dump(images, open(PATH_OUT + name_album_file, 'wb'))
        # print('Writing: ' + name_album_file)
    # else:
    #     print('The file already exists: %s' % name_album_file)
# ___________________________________________________________________________


def main():
    name_album_files = os.listdir(PATH_DATA)
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
    func = partial(process_album, lock, ns)
    processes.map(func, name_album_files)
    print('The image id retrieval is finished. (%d/%d)'
          % (ns.counter, ns.counter_total))


if __name__ == '__main__':
    main()
