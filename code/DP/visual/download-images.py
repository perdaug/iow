
"""
VERSION
- Python 3

FUNCTION
- Download images from urls.
"""

from urllib.request import urlretrieve
import os
import multiprocessing
from optparse import OptionParser
import pandas as pd
import socket

socket.setdefaulttimeout(10)


op = OptionParser()
op.add_option('--clss', action='store', type=str,
              help='The image class (e.g., vietnam war).')
op.add_option('--source', action='store', type=str,
              help='The media source.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_DATA = PATH_HOME + '/data/IR/' + opts.source \
    + '/urls_image/' + opts.clss + '/'
PATH_OUT = PATH_HOME + '/data/DP/vision/' + opts.source + '/' \
    + opts.clss + '/'
N_THREADS = 16

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def download(lock, ns, info_image):
    try:
        lock.acquire()
        ns.counter_total += 1
        lock.release()
        name_image = info_image['name']
        url_image = info_image['url']
        if name_image not in os.listdir(PATH_OUT):
            urlretrieve(url_image, PATH_OUT + name_image)
            print('Downloaded an image: %s' % name_image)
            lock.acquire()
            ns.counter += 1
            lock.release()
        else:
            # print('File already exists: %s' % name_image)
            return
    except Exception as e:
        print(e)
        return
# ___________________________________________________________________________


def main():
    infos_image = []
    for name_file in os.listdir(PATH_DATA):
        info_image = pd.read_pickle(PATH_DATA + name_file)
        infos_image.append(info_image)
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
    func = partial(download, lock, ns)
    processes.map(func, infos_image)
    print('The image retrieval is finished (%d/%d).'
          % (ns.counter, ns.counter_total))


if __name__ == '__main__':
    main()
