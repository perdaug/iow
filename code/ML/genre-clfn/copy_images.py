import os
from shutil import copyfile
from optparse import OptionParser
import glob
import pandas as pd

op = OptionParser()
op.add_option("--model", action="store", type=str,
		help="The name of the applied model.")
op.add_option("--corpus", action="store", type=str,
		help="A particular collection of images.")
op.add_option("--source", action="store", type=str,
		help="The origin of a collection of images.")
(opts, args) = op.parse_args()

PATH_HOME = os.path.expanduser('~') + '/Projects/iow'
PATH_DATA = PATH_HOME + '/data/genre-clfn/paths_image-clusters/' \
		+ opts.model + '/' + opts.corpus + '/' + opts.source + '/'
PATH_OUT = PATH_HOME + '/data/genre-clfn/images_copied/' \
		+ opts.model + '/' + opts.corpus + '/' + opts.source + '/'

pickles_path = glob.glob(PATH_DATA + '*')

for pickle_path in pickles_path:
	tuples_img = pd.read_pickle(pickle_path)
	for tuple_img in tuples_img:
		name_dir = tuple_img[1].split('/')[-1]
		if not os.path.exists(PATH_OUT + name_dir):
			os.makedirs(PATH_OUT + name_dir)
		path_img = tuple_img[0]
		name_img = path_img.split('/')[-1]
		copyfile(path_img, PATH_OUT + name_dir + '/' + name_img)
