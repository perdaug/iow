
"""
VERSION
- Python 2

FUNCTION
- Retrieve imagenet images from their class ids

IDS
Trench: n04242315
Death camp: n03166685
Drone: n03245889
Memorial, monument: n03743902
Prison camp: n04005912
Weapon: n04565375
Fighter aircraft: n03335030
Politician: n10450303
Capital ship: n02956393
Rifleman: n10530571
Tank: n04389033
Face: n09618957
People: n07942152
Warship: n04552696
field: n09393605
"""

import os
import urllib
import urllib2
from optparse import OptionParser
import numpy as np
import multiprocessing

op = OptionParser()
op.add_option('--wnid', action='store', type=str,
              help='The query ID.')
op.add_option('--query', action='store', type=str,
              help='The name the class.')
op.add_option('-n', action='store', type=int,
              help='The image count threshold.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow/'
PATH_OUT = PATH_HOME + 'data/DP/vision/imagenet/' + opts.query + '/'
N_THREADS = 16

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)
# ___________________________________________________________________________


def retrieve_image(lock, ns, len_urls, url):
    try:
        if (ns.counter >= opts.n):
            raise Exception('Reached the image limit.')
        lock.acquire()
        ns.counter_total += 1
        print('Starting the attempt (%d/%d).'
              % (ns.counter_total, len_urls))
        lock.release()
        name_img = url.split('/')[-1]
        if name_img in os.listdir(PATH_OUT):
            raise Exception('The file already exists: %s' % name_img)
        header_img = urllib2.urlopen(url).info()
        header_content = header_img.getheader('Content-Type')
        if header_content == 'image/jpeg':
            urllib.urlretrieve(url, PATH_OUT + name_img)
            print('Images retrieved: %d' % ns.counter)
            lock.acquire()
            ns.counter += 1
            lock.release()
    except Exception as e:
        print(e)

# ___________________________________________________________________________


def main():
    '''
    Extracting the urls
    '''
    query_url = 'http://www.image-net.org/api/text/' \
                + 'imagenet.synset.geturls?wnid=%s' % opts.wnid
    response = urllib2.urlopen(query_url).read()
    urls = response.split('\r\n')
    urls = np.array(urls)
    np.random.shuffle(urls)
    '''
    Enabling multiprocessing
    '''
    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    ns.counter = 0
    ns.counter_total = 0
    lock = manager.Lock()
    from functools import partial
    func = partial(retrieve_image, lock, ns, len(urls))
    processes = multiprocessing.Pool(N_THREADS)
    processes.map(func, urls)
    processes.close()


if __name__ == '__main__':
    main()
