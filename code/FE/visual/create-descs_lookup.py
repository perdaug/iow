
"""
VERSION
- Python 2
- OpenCV

FUNCTION
- Create the SIFT descriptors for a corpus
"""

import os
import glob
import visual_bow as bow
import cPickle as pkl
import cv2
from optparse import OptionParser

op = OptionParser()
op.add_option('--lookup', action='store', type=str,
              help='The data to be examined.')
op.add_option('--source', action='store', type=str,
              help='The origin of a collection of images.')
op.add_option('-n', action='store', type=int,
              help='The number of elements per partition.')
op.add_option('--scale', action='store', type=str,
              help='The image scale.')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_IMAGE = PATH_HOME + '/data/DP/vision/' + opts.source \
    + '/' + opts.lookup + '/'
PATH_OUT = PATH_HOME + '/data/FE/visual/' + opts.source \
    + '/descs_' + opts.scale + '/' + opts.lookup + '/'
if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)

# ___________________________________________________________________________


def create_desc_sift(path_image):
    try:
        img = bow.read_image(path_image)
        if int(opts.scale) != 0:
            img = cv2.resize(img, (int(opts.scale), int(opts.scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(gray, None)
        return desc
    except:
        return None


def dump_partition(descs, ctr, n_partitions):
    name_partition = opts.lookup + '_' + str(ctr) + '_' \
        + str(opts.scale) + '.pkl'
    with open(PATH_OUT + name_partition, 'wb') as file_out:
        pkl.dump(descs, file_out)
        print('Dumped a partition. (%s/%s)' % (ctr, n_partitions))
# ___________________________________________________________________________


def main():
    paths_images = glob.glob(PATH_IMAGE + '*')
    descs = []
    n_partitions = len(paths_images) / opts.n + 1
    ctr = 1
    '''
    Creating SIFT descriptors in batches.
    '''
    for path_image in paths_images:
        desc = create_desc_sift(path_image)
        if desc is not None:
            descs.append((desc, path_image))
        if len(descs) == opts.n:
            dump_partition(descs, ctr, n_partitions)
            descs = []
            ctr += 1
    '''
    Dumping the remainder batch.
    '''
    dump_partition(descs, ctr, n_partitions)


if __name__ == '__main__':
    main()
