
"""
VERSION
- Python 3

FUNCTION
- Write image ids to a file
"""

from optparse import OptionParser
import requests
import os
import pandas as pd
import pickle as pkl

op = OptionParser()
op.add_option('--query', action='store', type=str,
              help='The query submitted to Flickr.')
op.add_option('--clss', action='store', type=str,
              help='The image class (e.g., vietnam war).')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow/'
PATH_OUT = PATH_HOME + 'data/IR/flickr/ids_image/'
KEY_API = ''
NAME_OUT_FILE = 'ids_' + opts.clss + '.pkl'
THRESHOLD_EMPTY = 5

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def main():
    dict_class = {}
    if os.path.exists(PATH_OUT + NAME_OUT_FILE):
        dict_class = pd.read_pickle(PATH_OUT + NAME_OUT_FILE)
    else:
        dict_class = {'data': {}, 'query': {}}
    if opts.query not in dict_class['query']:
        dict_class['query'][opts.query] = 0
    dict_class['query'][opts.query] += 1
    len_last = len(dict_class['data'])

    counter_empty = 0
    '''
    Obtaining the number of pages
    '''
    url_n_pages = 'https://api.flickr.com/services/rest/' \
                  '?method=flickr.photos.search' \
                  '&per_page=500&format=json&nojsoncallback=?' \
                  '&api_key=%s&text=%s&page=%s' \
                  % (KEY_API, opts.query.replace('-', ' '), 1)
    response = requests.get(url_n_pages)
    json_response = response.json()
    n_pages = json_response['photos']['pages']

    '''
    - Sending the image request;
    - Extract the image id.
    '''
    idxs_page = list(range(0, n_pages))
    for idx_page in idxs_page:
        url_image_batch = 'https://api.flickr.com/services/rest/' \
                          '?method=flickr.photos.search' \
                          '&per_page=500&format=json&nojsoncallback=?' \
                          '&api_key=%s&text=%s&page=%s' \
                          % (KEY_API, opts.query.replace('-', ' '), idx_page)
        response = requests.get(url_image_batch)
        json_response = response.json()
        for json_image in json_response['photos']['photo']:
            id_image = json_image['id']
            if id_image not in dict_class['data']:
                dict_class['data'][id_image] = {}
            if opts.clss not in dict_class['data'][id_image]:
                dict_class['data'][id_image][opts.clss] = 0
            dict_class['data'][id_image][opts.clss] += 1
        if len(dict_class['data']) == len_last:
            counter_empty += 1
        if counter_empty == THRESHOLD_EMPTY:
            break
        len_last = len(dict_class['data'])
        print('The number of images in iteration %d: %d'
              % (idx_page, len_last))
    '''
    Dumping the dictionary with image ids.
    '''
    pkl.dump(dict_class, open(PATH_OUT + NAME_OUT_FILE, 'wb'))
    print('The total number of images in %s: %s' % (opts.clss, len_last))


if __name__ == '__main__':
    main()
