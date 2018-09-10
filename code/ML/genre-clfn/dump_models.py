import os
import glob
import visual_bow as bow
import cPickle as pkl
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from time import time
from optparse import OptionParser

t0 = time()

op = OptionParser()
op.add_option("--model", action="store", type=str, \
        help="The model name.")
(opts, args) = op.parse_args()

HOME_PATH = os.path.expanduser('~') + '/Projects/iow'
IMAGE_PATH = HOME_PATH + '/data/ML/genre-clfn/descs_training/' \
        + opts.model + '/'
MODELS_PATH = HOME_PATH + '/data/ML/genre-clfn/models_trained/'

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
# ________________________________________________________________

# PRE-PROCESSING TRAINING DATA
pickle_descs = glob.glob(IMAGE_PATH + '*')
descs_train = []
y_train = []
for path_partition in pickle_descs:
    print('Processed partition: %s' % path_partition.split('/')[-1])
    pickle_desc = pkl.load(open(path_partition, 'rb'))
    for desc in pickle_desc:
        if desc[1] is None or desc[0] is None:
            continue
        descs_train.append(desc[0])
        print(desc[1])
        y_train.append(desc[1])
training_idxs = []
for i in range(len(descs_train)):
    training_idxs.append(i)

# INITIALISING AN SVM
c_vals = [0.01, 0.1, 1, 10, 100, 1000]
gamma_vals = [0.001, 0.0001, 0.00001, 0.1, 1, 100]
param_grid = [
    {'C': c_vals, 'kernel': ['linear']},
    {'C': c_vals, 'gamma': gamma_vals, 'kernel': ['rbf']},
 ]
SCORING = 'precision_micro'
svc = GridSearchCV(SVC(), param_grid, n_jobs=-1, scoring=SCORING)

# CLUSTERING THE DATA
K_CLUSTERS = 700
X_train, cluster_model = bow.cluster_features(
    descs_train, training_idxs=training_idxs,
    cluster_model=MiniBatchKMeans(n_clusters=K_CLUSTERS)
)
print('The number of training images: %s' % len(X_train))
# ________________________________________________________________
# CLASSIFICATION
# TODO: TF-IDF TRANSFORMATION
print(len(pickle_descs))
svc.fit(X_train, y_train)

name_model = opts.model + '_k-means.pkl'
pkl.dump(cluster_model, open(MODELS_PATH + name_model, 'wb'))

name_svc_model = opts.model + '_svc.pkl'
pkl.dump(svc, open(MODELS_PATH + name_svc_model, 'wb'))

print('Script runtime: %.2f' % (time() - t0))
