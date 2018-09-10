import os
import glob
import visual_bow as bow
import pandas as pd
import numpy as np
from time import time
from optparse import OptionParser

t0 = time()

op = OptionParser()
op.add_option("--lookup", action="store", type=str,
		help="A particular collection of images.")
op.add_option("--source", action="store", type=str,
		help="The primary source of images.")
op.add_option("--model", action="store", type=str,
		help="The applied model.")
op.add_option("--scale", action="store", type=str,
		help="The image scale.")
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_LOOKUP = PATH_HOME + '/data/corpora/' + opts.source \
		+ '/descs_lookup/' + opts.scale + '/' + opts.lookup + '/'
PATH_MODEL = PATH_HOME + '/data/genre-clfn/models_trained/'
PATH_LEXICON = PATH_HOME + '/data/lexicons/imagenet/images_raw/'
PATH_OUT = PATH_HOME + '/data/genre-clfn/paths_image-clusters/' \
		+ opts.model + '/' + opts.lookup + '/' + opts.source + '/'
if not os.path.exists(PATH_OUT):
	os.makedirs(PATH_OUT)
# ________________________________________________________________
kmeans = pd.read_pickle(PATH_MODEL + opts.model + '_k-means.pkl')
svc = pd.read_pickle(PATH_MODEL + opts.model + '_svc.pkl')
pickles_lookup = glob.glob(PATH_LOOKUP + '*')
for idx, pickle_lookup in enumerate(pickles_lookup):

	# READ THE DATA
	descs = pd.read_pickle(pickle_lookup)
	descs_lookup = []
	paths_lookup = []
	for desc in descs:
		if desc[1] is None or desc[0] is None:
			continue
		descs_lookup.append(desc[0])
		paths_lookup.append(desc[1])
	paths_lookup = np.array(paths_lookup)

	# CREATE VISUAL WORDS
	# TODO: IMPLEMENT TF-IDF
	X_lookup, model = bow.cluster_features(descs_lookup, \
			training_idxs=[], cluster_model=kmeans)

	# ASSIGN THE CLASS
	y_lookup = svc.predict(X_lookup)
	# PRINTING THE DIRS OF INTEREST
	dirs_target = glob.glob(PATH_LEXICON + '*')
	print dirs_target
	idxs_all = []
	dirs_all = []
	for dir_target in dirs_target:
		idxs_target = np.where(y_lookup == dir_target)
		for idx_target in idxs_target[0]:
			idxs_all.append(idx_target)
			name_dir = dir_target.split('/')[-1]
			dirs_all.append(name_dir)
	print(np.array(zip(paths_lookup[idxs_all], dirs_all)))
	print('Iteration (%d/%d). Time taken: %.2f' \
			  % (idx + 1, len(pickles_lookup), time() - t0))
	print('The number of images matching the classes: %s' \
				% len(idxs_all))

	# WRITING THE RESULTS
 	paths_out = np.array(zip(paths_lookup, y_lookup))
	paths_out.dump(PATH_OUT + opts.model + '_' + str(idx) + '.pkl')
