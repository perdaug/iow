
"""
VERSION
- Python 2
- OpenCV

FUNCTION
- Create the SIFT descriptors for the targets
"""

import os
import glob
import visual_bow as bow
import cPickle as pkl
import cv2
from optparse import OptionParser

op = OptionParser()
op.add_option('--scale', action='store', type=str,
              help='The image scale.')
op.add_option('--target', action='store', type=str,
              help='The target data set (claim).')
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_TARGET = PATH_HOME + '/data/DP/vision/imagenet/selected/' \
    + opts.target + '/'
PATH_OUT = PATH_HOME + '/data/FE/visual/imagenet/' + opts.target + '/' \
    + 'descs_' + str(opts.scale) + '/'
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
    except Exception as e:
        print(e, path_image)
        return None


def dump_target_class(descs, idx_target, target, paths_targets):
    name_target_class = target.split('/')[-1]
    # if not os.path.exists(PATH_OUT + name_target_class):
    #     os.makedirs(PATH_OUT + name_target_class)
    name_out_file = name_target_class + '_' + str(len(descs)) + '_' \
        + str(opts.scale) + '.pkl'
    with open(PATH_OUT + name_out_file, 'wb') as f:
    # with open(PATH_OUT + name_target_class + '/' + name_out_file, 'wb') as f:
        pkl.dump(descs, f)
    print('Finished iterating class %s. (%d/%d)'
          % (target, idx_target+1, len(paths_targets)))
# ___________________________________________________________________________


def main():
    if not os.path.exists(PATH_OUT):
        os.makedirs(PATH_OUT)
    paths_targets = glob.glob(PATH_TARGET + '*')
    for idx_target, target in enumerate(paths_targets):
        descs = []
        paths_target_images = glob.glob(target + '/*')
        for idx_image, path_image in enumerate(paths_target_images):
            desc = create_desc_sift(path_image)
            if desc is not None:
                # descs.append((desc, target, path_image))
                descs.append((desc, path_image, target))
        dump_target_class(descs, idx_target, target, paths_targets)


if __name__ == '__main__':
    main()
