
"""
VERSION
- Python 3

FUNCTION
- Download the albums ids given the query.

DEPENDENCIES
- http://facebook-sdk.readthedocs.io/en/latest/install.html
"""

import os
import multiprocessing
import facebook
import requests
from optparse import OptionParser
import pickle as pkl

op = OptionParser()
op.add_option('--query',
              action='store', type=str,
              help='Picture class.')
(opts, args) = op.parse_args()

N_THREADS = 16
ACCESS_TOKEN = ''
PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_OUT = PATH_HOME + '/data/IR/fb/albums/raw/' + opts.query + '/'

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def process_page(lock, ns, id_page):
    try:
        lock.acquire()
        ns.counter_total += 1
        print('Processing: ' + id_page)
        lock.release()
        if os.path.exists(PATH_OUT + '/' + id_page + '.pkl'):
            print('Exit: The file already exists: %s' % (id_page))
            return
        '''
        Extracting the albums.
        '''
        url_album_batch = 'https://graph.facebook.com/' \
            + '%s/?fields=albums&access_token=%s' \
            % (id_page, ACCESS_TOKEN)
        response_album_batch = requests.get(url_album_batch)
        batch_album = response_album_batch.json()['albums']
        page_albums = []
        page_albums.append(batch_album['data'])
        while 'next' in batch_album['paging']:
            url_album_batch = batch_album['paging']['next']
            batch_album = requests.get(url_album_batch).json()
            page_albums.append(batch_album['data'])
        '''
        Writing the album data.
        '''
        # print('Writing: ' + id_page)
        pkl.dump(page_albums, open(PATH_OUT + id_page + '.pkl', 'wb'))
        lock.acquire()
        ns.counter += 1
        lock.release()
    except Exception as e:
        # print(e)
        return
# ___________________________________________________________________________


def main():
    retriever_fb = facebook.GraphAPI(ACCESS_TOKEN)
    query = opts.query.replace('-', ' ')
    '''
    Extracting the page ids.
    '''
    request_pages = retriever_fb.request('search',
                                         {'q': query, 'type': 'page'})
    ids_page = []
    while len(request_pages['data']) > 0:
        for datum_page in request_pages['data']:
            ids_page.append(datum_page['id'])
        id_next_page = request_pages['paging']['cursors']['after']
        request_pages = retriever_fb.request('search',
                                             {'q': query, 'type': 'page',
                                              'after': id_next_page})
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
    func = partial(process_page, lock, ns)
    processes.map(func, ids_page)
    print('The album retrieval is finished (%d/%d).'
          % (ns.counter, ns.counter_total))


if __name__ == '__main__':
    main()
