
"""
VERSION
- Python 2
- OpenCV

FUNCTION
- Ranking the look-up images based on the SIFT descriptors.
"""

import os
import glob
import visual_bow as bow
import cPickle as pkl
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial import distance
from time import time
from optparse import OptionParser
from shutil import copyfile
import numpy as np

t0 = time()

op = OptionParser()
op.add_option('--lookup', action='store', type=str,
              help='The lookup directory.')
op.add_option('--scale', action='store', type=str,
              help='The image scale.')
op.add_option('--size', action='store', type=int,
              help='The number of results.')
op.add_option('-K', action='store', type=int,
              help='The number of clusters.')
op.add_option('--target', action='store', type=str,
              help='The target data set.')
op.add_option('--source', action='store', type=str,
              help='The corpus source.')
(opts, args) = op.parse_args()

# TODO: CHANGE THE TARGET TO CLAIM (NAMING ISSUE)
PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_LOOKUP = PATH_HOME + '/data/FE/visual/' + opts.source + '/descs_' \
    + opts.scale + '/' + opts.lookup + '/'
PATH_TARGET = PATH_HOME + '/data/FE/visual/imagenet/' + opts.target + '/' \
    + 'descs_' + opts.scale + '/'
PATH_OUT = PATH_HOME + '/data/ML/image-search/SIFT-%s_K-%s/'  \
    % (opts.scale, opts.K) + opts.source + '/' + opts.lookup + '/' \
    + opts.target + '/'
# ___________________________________________________________________________
# TODO: LOOK INTO ALTERNATIVE DISTANCE METRICS


class Searcher:
    def __init__(self, index, paths):
        self.index = index
        self.paths = paths

    def search(self, queryFeatures):
        results = {}
        for idx, lookup in enumerate(self.index):
            d = distance.euclidean(lookup, queryFeatures)
            key = self.paths[idx]
            results[key] = d
        results = sorted([(v, k) for (k, v) in results.items()])
        return results
# ___________________________________________________________________________


def main():
    pickles_lookup = glob.glob(PATH_LOOKUP + '*')
    paths_target = glob.glob(PATH_TARGET + '*')
    cluster_model = MiniBatchKMeans(n_clusters=opts.K)
    descs_sift = []
    paths_all = []
    '''
    Reading the look-up descriptors,
    '''
    for pickle_lookup in pickles_lookup:
        name_lookup_partition = pickle_lookup.split('/')[-1]
        print('Processing a look-up partition: %s' % name_lookup_partition)
        descs_lookup = pkl.load(open(pickle_lookup, 'rb'))
        for desc in descs_lookup:
            if desc[0] is None or desc[1] is None:
                continue
            descs_sift.append(desc[0])
            paths_all.append(desc[1])
    '''
    Reading the target descriptors.
    '''
    idxs_target = []
    for path_trgt in paths_target:
        # pickles_target = glob.glob(path_trgt + '/*')
        # print(pickles_target)
        # for pickle_target in pickles_target:
        print('Processing a target partition: %s'
              % path_trgt.split('/')[-1])
        descs_lookup = pkl.load(open(path_trgt, 'rb'))
        for desc in descs_lookup:
            if desc[0] is None or desc[1] is None:
                continue
            idx_target = len(descs_sift)
            idxs_target.append(idx_target)
            descs_sift.append(desc[0])
            paths_all.append(desc[1])

    '''
    Clustering the descriptors.
    '''
    idxs_all = []
    for i in range(len(descs_sift)):
        idxs_all.append(i)
    X, cluster_model = bow.cluster_features(descs_sift,
                                            training_idxs=idxs_all,
                                            cluster_model=cluster_model)
    print('The number of clustered images: %s' % len(X))
    '''
    Pre-processing the clustered descriptors.
    '''
    print(idxs_target)
    transformer = TfidfTransformer(smooth_idf=False)
    X = transformer.fit_transform(X).toarray()
    X_target = X[idxs_target]
    X_lookup = np.delete(X, idxs_target, axis=0)
    paths_all = np.array(paths_all)
    paths_lookup = np.delete(paths_all, idxs_target, axis=0)
    '''
    Searching for the best matches.
    '''
    searcher = Searcher(X_lookup, paths_lookup)
    for idx, target in enumerate(X_target):
        results = searcher.search(target)
        idx_target = idxs_target[idx]
        path_trgt = paths_all[idx_target]
        dir_target = path_trgt.split('/')[-2] + '/'
        dir_target_img = path_trgt.split('/')[-1][:-4] + '/'
        name_target_img = path_trgt.split('/')[-1]
        if not os.path.exists(PATH_OUT + dir_target + dir_target_img):
            os.makedirs(PATH_OUT + dir_target + dir_target_img)
        '''
        Copying the target as the 0th element.
        '''
        copyfile(path_trgt, PATH_OUT + dir_target + dir_target_img +
                 '0_' + name_target_img)
        print('Output path: %s' % PATH_OUT + dir_target + dir_target_img)
        print('Target image data: %s' % path_trgt)
        '''
        Copying the closest matches.
        '''
        for j in range(0, opts.size):
            score = results[j]
            path_img, score_img = score[1], score[0]
            name_img = str(j + 1) + '_' + path_img.split('/')[-1]
            print('\t%d. %s : %.3f' % (j + 1, path_img, score_img))
            copyfile(path_img, PATH_OUT + dir_target + dir_target_img +
                     name_img)
    print('Script run-time: %.2f' % (time() - t0))


if __name__ == '__main__':
    main()
