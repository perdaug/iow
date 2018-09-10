
"""
VERSION
- Python 2
- OpenCV

FUNCTION
- Ranking the look-up images based on the colour descriptors.
"""

import os
import glob
import numpy as np
import cv2
from time import time
from optparse import OptionParser
from shutil import copyfile

op = OptionParser()
op.add_option('--lookup', action='store', type=str,
              help='The lookup directory.')
op.add_option('--size', action='store', type=int,
              help='The number of results.')
op.add_option('--target', action='store', type=str,
              help='The target data set.')
op.add_option('--source', action='store', type=str,
              help='The corpus source.')
(opts, args) = op.parse_args()

# TODO: CHANGE THE TARGET TO CLAIM (NAMING ISSUE)
PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_LOOKUP = PATH_HOME + '/data/DP/vision/' + opts.source + '/' \
    + opts.lookup + '/'
PATH_TARGET = PATH_HOME + '/data/DP/vision/imagenet/selected/' \
    + opts.target + '/'
PATH_OUT = PATH_HOME + '/data/ML/image-search/colour/' \
    + opts.source + '/' + opts.lookup + '/' + opts.target + '/'
# ___________________________________________________________________________


class Searcher:
    def __init__(self, index):
        self.index = index

    def search(self, features_target):
        results = {}
        for (k, features) in self.index.items():
            d = self.chi2_distance(features, features_target)
            results[k] = d
        results = sorted([(v, k) for (k, v) in results.items()])
        return results

    def chi2_distance(self, hist_a, hist_b, eps=1e-10):
        d = 0
        for (a, b) in zip(hist_a, hist_b):
            d += ((a - b)**2) / (a + b + eps)
        d *= 0.5
        return d
# ___________________________________________________________________________
# TODO: Reference the source of the script


class HSV_Histogram:
    def __init__(self, bins):
        self.bins = bins

    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        '''
        Creating the masks.
        '''
        (h, w) = image.shape[:2]
        (center_x, center_y) = (int(w * 0.5), int(h * 0.5))
        segments = [(0, center_x, 0, center_y), (center_x, w, 0, center_y),
                    (center_x, w, center_y, h), (0, center_x, center_y, h)]
        (axes_x, axes_y) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
        mask_ellipse = np.zeros(image.shape[:2], dtype='uint8')
        cv2.ellipse(mask_ellipse, (center_x, center_y), (axes_x, axes_y),
                    0, 0, 360, 255, -1)
        '''
        Extracting the histograms related to the masks.
        '''
        features = []
        for (start_x, end_x, start_y, end_y) in segments:
            mask_corner = np.zeros(image.shape[:2], dtype='uint8')
            cv2.rectangle(mask_corner, (start_x, start_y),
                          (end_x, end_y), 255, -1)
            mask_corner = cv2.subtract(mask_corner, mask_ellipse)
            hist = self.histogram(image, mask_corner)
            features.extend(hist)
        hist = self.histogram(image, mask_ellipse)
        features.extend(hist)
        return features

# ___________________________________________________________________________
# NOTE: BACKLOG


class RGB_HISTOGRAM:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins,
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist)
        return hist.flatten()

# ___________________________________________________________________________
# TODO: Dedicate a flag to choose the colour descriptor


def main():
    t0 = time()
    '''
    Creating the features of the target.
    '''
    # factory_rgb = RGB_HISTOGRAM([4, 4, 4])
    factory_hsv = HSV_Histogram((4, 6, 2))
    paths_target = glob.glob(PATH_TARGET + '*')
    dict_target_features = {}
    for path_target in paths_target:
        for path_target_image in glob.glob(path_target + '/*'):
            try:
                image = cv2.imread(path_target_image)
                # feature_rgb = factory_rgb.describe(image)
                feature_hsv = factory_hsv.describe(image)
                # dict_target_features[path_target_image] = feature_rgb
                dict_target_features[path_target_image] = feature_hsv
            except Exception as e:
                continue
    print('Indexed the templates. Time taken: %.2f' % (time() - t0))
    '''
    Creating the features of the look-up.
    '''
    t0 = time()
    paths_lookup_images = glob.glob(PATH_LOOKUP + '*')
    dict_lookup_features = {}
    for path_training_image in paths_lookup_images:
        try:
            image = cv2.imread(path_training_image)
            # feature_rgb = factory_rgb.describe(image)
            feature_hsv = factory_hsv.describe(image)
        except:
            continue
        # dict_lookup_features[path_training_image] = feature_rgb
        dict_lookup_features[path_training_image] = feature_hsv
    print('Indexed the look-up data. Time taken: %.2f' % (time() - t0))
    '''
    Identifying the closest look-up matches.
    '''
    t0 = time()
    searcher = Searcher(dict_lookup_features)
    for (path_target, features_target) in dict_target_features.items():
        results = searcher.search(features_target)
        name_target = path_target.split('/')[-2]
        name_target_image = path_target.split('/')[-1][:-4]
        path_out_match = PATH_OUT + name_target + '/' + name_target_image \
            + '/'
        if not os.path.exists(path_out_match):
            os.makedirs(path_out_match)
        '''
        Copying the target.
        '''
        print('The target path: %s' % (path_target))
        copyfile(path_target, path_out_match + '0_' +
                 name_target_image + '.jpg')
        '''
        Copying the closest matches.
        '''
        for j in xrange(1, opts.size):
            (score, path_image) = results[j]
            name_image = str(j) + '_' + path_image.split('/')[-1]
            print('\t%d. %s : %.3f' % (j + 1, path_image, score))
            copyfile(path_image, path_out_match + name_image)
    print('The queries are finished. Time taken: %.2f' % (time() - t0))


if __name__ == '__main__':
    main()
