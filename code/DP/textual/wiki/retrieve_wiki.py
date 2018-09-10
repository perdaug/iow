
"""
VERSION
- Python 3

FUNCTION
- Retrieve the content from wikipedia

DEPENDENCIES
- pip install wikipedia
"""

import os
import multiprocessing
import wikipedia
from optparse import OptionParser
import pickle as pkl

op = OptionParser()
op.add_option('--query', action='store', type=str,
              help='The topic of the lexicon.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow/'
PATH_OUT = PATH_HOME + '/data/DP/text/wiki/' + opts.query + '/'
N_THREADS = 16
N_QUERIES = 2500

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def process_page(name_page):
    try:
        data_page = wikipedia.page(name_page)
        content_page = data_page.content
        name_page = str(data_page.revision_id) + '.pkl'
        pkl.dump(content_page, open(PATH_OUT + name_page, 'wb'))
    except Exception as e:
        print(e)
        return
# ___________________________________________________________________________


def main():
    names_page = wikipedia.search(opts.query, results=N_QUERIES)
    '''
    Enabling multiprocessing
    '''
    processes = multiprocessing.Pool(N_THREADS)
    processes.map(process_page, names_page)
    print('Total pages: %d.' % len(names_page))


if __name__ == '__main__':
    main()
