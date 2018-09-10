import os
import glob
import visual_bow as bow
import cPickle as pkl
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from optparse import OptionParser

np.set_printoptions(threshold=np.nan)

op = OptionParser()
op.add_option('--training', action='store', type=str,
              help='The training corpora.')
(opts, args) = op.parse_args()

HOME_PATH = os.path.expanduser('~') + '/Projects/iow/'
# IMAGE_PATH = HOME_PATH + '/data/genre-clfn/descs_training/' \
#     + opts.training + '/'
IMAGE_PATH = HOME_PATH + 'data/FE/visual/imagenet/genres/descs_128/'
# ________________________________________________________________


def run_svm(X_train, X_test, y_train, y_test, scoring):
    # c_vals = [1, 10, 100, 1000]
    # gamma_vals = [0.0001, 0.00001]
    c_vals = [0.01, 1, 100, 1000, 10000]
    gamma_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_grid = [
        {'C': c_vals, 'gamma': gamma_vals, 'kernel': ['rbf']},
        {'C': c_vals, 'degree': [2, 3, 4, 5], 'kernel': ['poly']},
        {'kernel': ['linear'], 'C': c_vals}]

    svc = GridSearchCV(SVC(), param_grid, n_jobs=-1, scoring=scoring)
    svc.fit(X_train, y_train)
    print 'train score (%s):' % scoring, svc.score(X_train, y_train)
    test_score = svc.score(X_test, y_test)
    print 'test score (%s):' % scoring, test_score

    print svc.best_params_
    print svc.best_score_

    return svc, test_score


def cluster_and_split(img_descs, y, training_idxs, test_idxs, \
        val_idxs, K):
    X, cluster_model = bow.cluster_features(img_descs, \
            training_idxs=training_idxs, \
            cluster_model=MiniBatchKMeans(n_clusters=K)
    )
    # TODO: TF-IDF TRANSFORMATION
    transformer = TfidfTransformer(smooth_idf=False)

    # X = transformer.fit_transform(X).toarray()

    X_train, X_test, X_val, y_train, y_test, y_val = bow.perform_data_split(X, y, training_idxs, test_idxs, val_idxs)

    return X_train, X_test, X_val, y_train, y_test, y_val, cluster_model

def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
# ________________________________________________________________
# READING TRAINING DATA


pickle_descs = glob.glob(IMAGE_PATH + '*')
descs = []
y = []
for pickle_dsc in pickle_descs:
    path_partitions = glob.glob(pickle_dsc + '/*')
    for path_partition in path_partitions:
        pickle_desc = pd.read_pickle(path_partition)
        for desc in pickle_desc:
            if desc[1] is None or desc[0] is None:
                continue
            descs.append(desc[0])
            y.append(desc[1])
y = np.array(y)
descs = np.array(descs)
descs, y = unison_shuffled_copies(descs, y)
training_idxs, test_idxs, val_idxs = bow.train_test_val_split_idxs(
        total_rows=len(descs), percent_test=0.15, percent_val=0)
# ________________________________________________________________
# RUNNING THE EVALUATION

training_idxs = np.array(training_idxs)
scoring = 'precision_micro'
results = {}
# K_vals = [300, 500, 700]
K_vals = [300, 500, 1000]

for K in K_vals:
    X_train, X_test, X_val, y_train, y_test, y_val, \
        cluster_model = cluster_and_split(descs, y, \
        training_idxs, test_idxs, val_idxs, K)

    print('Inertia for clustering with K=%i is:' % K, cluster_model.inertia_)
    print('SVM Scores: ')
    svmGS, svm_score = run_svm(X_train, X_test, y_train, y_test, scoring)

    print '\n*** K=%i DONE ***\n' % K
